#!/usr/bin/env python3
import argparse
import gc
import pickle
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_SYSTEM_PROMPT = (
    "Solve the following math word problem. Show your reasoning, then give the final "
    "answer in the format '#### <answer>'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline rollouts with vLLM and extract hidden states.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-path", default="rollouts_data.pkl")
    parser.add_argument("--dataset-name", default="MathArena/aime_2026")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--question-column", default="problem")
    parser.add_argument("--answer-column", default="answer")
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
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"


def normalize_numeric_string(text: str) -> Optional[str]:
    if text is None:
        return None
    value = text.strip().replace(",", "")
    value = re.sub(r"^\$+", "", value)
    value = value.rstrip(".")
    if not value:
        return None
    if value.startswith("."):
        value = f"0{value}"
    if value.startswith("-."):
        value = value.replace("-.", "-0.", 1)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value) is None:
        return None
    if "." not in value:
        sign = ""
        digits = value
        if digits.startswith("-"):
            sign = "-"
            digits = digits[1:]
        digits = digits.lstrip("0") or "0"
        return f"{sign}{digits}"
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def extract_reference_answer(answer_text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", answer_text)
    if match:
        return normalize_numeric_string(match.group(1))
    matches = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", answer_text)
    if matches:
        return normalize_numeric_string(matches[-1])
    return None


def extract_predicted_answer(generated_text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", generated_text)
    if match:
        return normalize_numeric_string(match.group(1))
    matches = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", generated_text)
    if matches:
        return normalize_numeric_string(matches[-1])
    return None


def compute_reward(generated_text: str, answer_text: str) -> int:
    gold = extract_reference_answer(answer_text)
    pred = extract_predicted_answer(generated_text)
    return int(gold is not None and pred is not None and gold == pred)


def load_prompts(args: argparse.Namespace, tokenizer: AutoTokenizer) -> List[Dict]:
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    sample_size = min(args.num_prompts, len(dataset))
    chosen_indices = random.sample(range(len(dataset)), sample_size)
    prompts: List[Dict] = []
    for prompt_order, dataset_idx in enumerate(chosen_indices):
        sample = dataset[dataset_idx]
        prompt_text = str(sample[args.question_column])
        answer_text = str(sample[args.answer_column])
        prompts.append(
            {
                "prompt_id": prompt_order,
                "dataset_idx": int(dataset_idx),
                "prompt_text": prompt_text,
                "prompt_rendered": build_prompt(prompt_text, tokenizer),
                "gold_answer": extract_reference_answer(answer_text),
                "answer_text": answer_text,
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
                    "reward": compute_reward(completion.text, prompt_item["answer_text"]),
                }
            )

        rollout_records.append(
            {
                "prompt_id": prompt_item["prompt_id"],
                "dataset_idx": prompt_item["dataset_idx"],
                "prompt_text": prompt_item["prompt_text"],
                "prompt_rendered": prompt_item["prompt_rendered"],
                "gold_answer": prompt_item["gold_answer"],
                "answer_text": prompt_item["answer_text"],
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
