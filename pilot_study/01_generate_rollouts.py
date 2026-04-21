#!/usr/bin/env python3
import argparse
import gc
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from code_evals import DEFAULT_TESTCASE_ROOT, judge


DEFAULT_SYSTEM_PROMPT = (
    "You are solving a competitive programming problem. Think through the approach "
    "carefully, then output a complete C++ solution that can be compiled with "
    "g++ -O2 -std=gnu++2a. Put the final answer inside a ```cpp ... ``` block, "
    "and make the last fenced block in the response be the final code."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate programming-task rollouts with vLLM and extract hidden states.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-path", default="rollouts_data.pkl")
    parser.add_argument("--dataset-path", default="code_evals/LCB_dataset.csv")
    parser.add_argument("--question-column", default="text")
    parser.add_argument("--id-column", default="no")
    parser.add_argument("--testcase-root", default=str(DEFAULT_TESTCASE_ROOT))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--num-trajectories", type=int, default=64)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--prefix-token-index", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-vllm-sleep-mode",
        action="store_true",
        help="Disable vLLM sleep mode before switching to the transformers hidden-state pass.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small end-to-end test with 2 prompts, 4 trajectories, and shorter generations.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n"
        "Write the full C++ solution and wrap it in a final ```cpp``` block."
    )


def compute_reward(generated_text: str, problem_id: str, testcase_root: str) -> int:
    status = judge(problem_id, generated_text, testcase_root=testcase_root)
    return int(status == 1)


def load_prompts(args: argparse.Namespace, tokenizer: AutoTokenizer) -> List[Dict]:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {args.id_column, args.question_column}
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    records = dataset.to_dict(orient="records")
    sample_size = min(args.num_prompts, len(records))
    chosen_indices = random.sample(range(len(records)), sample_size)
    prompts: List[Dict] = []
    for prompt_order, dataset_idx in enumerate(chosen_indices):
        sample = records[dataset_idx]
        prompt_text = str(sample[args.question_column])
        problem_id = str(sample[args.id_column])
        prompts.append(
            {
                "prompt_id": prompt_order,
                "dataset_idx": int(dataset_idx),
                "problem_id": problem_id,
                "prompt_text": prompt_text,
                "prompt_rendered": build_prompt(prompt_text, tokenizer),
                "gold_answer": None,
            }
        )
    return prompts


def apply_smoke_test_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.num_prompts = min(args.num_prompts, 2)
    args.num_trajectories = min(args.num_trajectories, 4)
    args.hidden_batch_size = min(args.hidden_batch_size, 2)
    args.prefix_token_index = min(args.prefix_token_index, 16)
    args.max_new_tokens = min(args.max_new_tokens, 64)


def generate_rollouts_with_vllm(
    args: argparse.Namespace,
    prompts: List[Dict],
) -> List[Dict]:
    sampling_params = SamplingParams(
        n=args.num_trajectories,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_sleep_mode=not args.disable_vllm_sleep_mode,
    )

    rollout_records: List[Dict] = []
    for prompt_item in tqdm(prompts, desc="vllm prompts"):
        outputs = llm.generate([prompt_item["prompt_rendered"]], sampling_params)
        request_output = outputs[0]
        trajectories = []
        for traj_id, completion in enumerate(request_output.outputs):
            token_ids = list(completion.token_ids)
            trajectories.append(
                {
                    "traj_id": traj_id,
                    "full_text": completion.text,
                    "generated_token_ids": token_ids,
                    "reward": compute_reward(completion.text, prompt_item["problem_id"], args.testcase_root),
                }
            )

        rollout_records.append(
            {
                "prompt_id": prompt_item["prompt_id"],
                "dataset_idx": prompt_item["dataset_idx"],
                "problem_id": prompt_item["problem_id"],
                "prompt_text": prompt_item["prompt_text"],
                "prompt_rendered": prompt_item["prompt_rendered"],
                "gold_answer": prompt_item["gold_answer"],
                "trajectories": trajectories,
            }
        )

    if not args.disable_vllm_sleep_mode:
        llm.sleep(level=2)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rollout_records


def extract_hidden_states_with_transformers(
    args: argparse.Namespace,
    rollout_records: List[Dict],
    tokenizer: AutoTokenizer,
) -> List[Dict]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=get_torch_dtype(args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_device = next(model.parameters()).device

    results: List[Dict] = []
    for prompt_item in tqdm(rollout_records, desc="hidden-state prompts"):
        prompt_input_ids = tokenizer(prompt_item["prompt_rendered"], add_special_tokens=False)["input_ids"]

        trajectories = prompt_item["trajectories"]
        finalized_trajectories = []

        batch_starts = range(0, len(trajectories), args.hidden_batch_size)
        for batch_start in tqdm(
            batch_starts,
            total=(len(trajectories) + args.hidden_batch_size - 1) // args.hidden_batch_size,
            desc=f"prompt {prompt_item['prompt_id']} hidden batches",
            leave=False,
        ):
            batch = trajectories[batch_start : batch_start + args.hidden_batch_size]
            batch_input_ids: List[List[int]] = []
            prefix_lengths: List[int] = []

            for traj in batch:
                generated_ids = traj["generated_token_ids"]
                prefix_len = min(args.prefix_token_index, len(generated_ids))
                prefix_lengths.append(prefix_len)
                batch_input_ids.append(prompt_input_ids + generated_ids[:prefix_len])

            encoded = tokenizer.pad(
                {"input_ids": batch_input_ids},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            final_hidden = outputs.hidden_states[-1]
            attention_mask = encoded["attention_mask"]
            valid_lengths = attention_mask.sum(dim=1).tolist()

            for row_idx, traj in enumerate(batch):
                prefix_len = prefix_lengths[row_idx]
                if prefix_len == 0:
                    hidden_dim = final_hidden.shape[-1]
                    prefix_hidden_state = np.zeros(hidden_dim, dtype=np.float32)
                    prefix_text = ""
                else:
                    final_pos = int(valid_lengths[row_idx] - 1)
                    prefix_hidden_state = (
                        final_hidden[row_idx, final_pos, :].detach().float().cpu().numpy()
                    )
                    prefix_text = tokenizer.decode(
                        traj["generated_token_ids"][:prefix_len],
                        skip_special_tokens=True,
                    )

                finalized_trajectories.append(
                    {
                        "traj_id": traj["traj_id"],
                        "full_text": traj["full_text"],
                        "prefix_text": prefix_text,
                        "prefix_hidden_state": prefix_hidden_state,
                        "reward": traj["reward"],
                    }
                )

            del outputs, final_hidden, encoded, attention_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(
            {
                "prompt_id": prompt_item["prompt_id"],
                "prompt_text": prompt_item["prompt_text"],
                "dataset_idx": prompt_item["dataset_idx"],
                "problem_id": prompt_item.get("problem_id"),
                "gold_answer": prompt_item["gold_answer"],
                "trajectories": finalized_trajectories,
            }
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()
    apply_smoke_test_overrides(args)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    prompts = load_prompts(args, tokenizer)
    rollout_records = generate_rollouts_with_vllm(args, prompts)
    results = extract_hidden_states_with_transformers(args, rollout_records, tokenizer)

    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
