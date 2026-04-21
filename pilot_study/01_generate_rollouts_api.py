#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from openai import APIError, APIConnectionError, APITimeoutError, OpenAI, RateLimitError
import pandas as pd
from tqdm.auto import tqdm

from code_evals import DEFAULT_TESTCASE_ROOT, judge


DEFAULT_SYSTEM_PROMPT = (
    "You are solving a competitive programming problem. Think through the algorithm, "
    "edge cases, and complexity, then output a complete C++ solution that can be "
    "compiled with g++ -O2 -std=gnu++2a. Put the final answer inside a ```cpp ... ``` "
    "block, and make the last fenced block in the response be the final code."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate programming-task rollouts through the OpenRouter API.")
    parser.add_argument("--model-name", default="google/gemma-3-27b-it")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--base-url", default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--http-referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--x-title", default=os.environ.get("OPENROUTER_X_TITLE", "pilot-study"))
    parser.add_argument("--output-path", default="rollouts_data.pkl")
    parser.add_argument("--dataset-path", default="code_evals/LCB_dataset.csv")
    parser.add_argument("--question-column", default="text")
    parser.add_argument("--id-column", default="no")
    parser.add_argument("--testcase-root", default=str(DEFAULT_TESTCASE_ROOT))
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--num-trajectories", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-requests", type=int, default=2)
    parser.add_argument("--prefix-token-index", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh", "none"],
        default="medium",
    )
    parser.add_argument("--probe-reasoning", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)


def apply_smoke_test_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.num_prompts = min(args.num_prompts, 2)
    args.num_trajectories = min(args.num_trajectories, 4)
    args.max_output_tokens = min(args.max_output_tokens, 64)
    args.parallel_requests = min(args.parallel_requests, 2)
    args.prefix_token_index = min(args.prefix_token_index, 16)


def compute_reward(generated_text: str, problem_id: str, testcase_root: str) -> int:
    status = judge(problem_id, generated_text, testcase_root=testcase_root)
    return int(status == 1)


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_programming_dataset(args: argparse.Namespace) -> List[Dict]:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {args.id_column, args.question_column}
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return dataset.to_dict(orient="records")


def extract_retry_delay_seconds(error: Exception) -> Optional[float]:
    message = str(error)
    patterns = [
        r"Please retry in (\d+(?:\.\d+)?)s",
        r"retry after (\d+(?:\.\d+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def normalize_reasoning_text(reasoning) -> str:
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(reasoning)


def combine_reasoning_and_content(reasoning_text: str, content_text: str) -> str:
    reasoning_text = reasoning_text.strip()
    content_text = content_text.strip()
    if reasoning_text and content_text:
        return f"{reasoning_text}\n\n{content_text}"
    if reasoning_text:
        return reasoning_text
    return content_text


def atomic_pickle_dump(path: str, obj: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        with open(tmp.name, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp.name, target)


def load_existing_results(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def build_results_index(results: List[Dict]) -> Dict[int, Dict]:
    return {item["dataset_idx"]: item for item in results}


def call_openrouter(
    client: OpenAI,
    args: argparse.Namespace,
    prompt_text: str,
    seed: int,
) -> Dict[str, str]:
    last_error: Optional[Exception] = None
    for attempt in range(args.max_retries + 1):
        try:
            extra_body = None
            if args.reasoning_enabled:
                extra_body = {
                    "reasoning": {
                        "enabled": True,
                        "effort": args.reasoning_effort,
                    }
                }
            response = client.chat.completions.create(
                model=args.model_name,
                messages=build_messages(prompt_text),
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_output_tokens,
                seed=seed,
                extra_body=extra_body,
            )
            message = response.choices[0].message
            content = message.content
            reasoning = getattr(message, "reasoning", None)
            return {
                "content": content if isinstance(content, str) else str(content),
                "reasoning": normalize_reasoning_text(reasoning),
            }
        except RateLimitError as error:
            last_error = error
            if attempt >= args.max_retries:
                raise
            retry_delay = extract_retry_delay_seconds(error) or args.retry_backoff * (attempt + 1)
            time.sleep(retry_delay)
        except (APIError, APIConnectionError, APITimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= args.max_retries:
                raise
            time.sleep(args.retry_backoff * (attempt + 1))
    raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def make_prefix_text(full_text: str, prefix_token_index: int) -> str:
    tokens = full_text.split()
    return " ".join(tokens[:prefix_token_index])


def generate_trajectories_for_prompt(
    client: OpenAI,
    args: argparse.Namespace,
    prompt_item: Dict,
    existing_prompt_result: Optional[Dict] = None,
    save_callback: Optional[Callable[[], None]] = None,
) -> List[Dict]:
    trajectories: List[Optional[Dict]] = [None] * args.num_trajectories
    completed_traj_ids = set()

    if existing_prompt_result:
        for traj in existing_prompt_result.get("trajectories", []):
            traj_id = traj.get("traj_id")
            if isinstance(traj_id, int) and 0 <= traj_id < args.num_trajectories:
                trajectories[traj_id] = traj
                completed_traj_ids.add(traj_id)

    pending_traj_ids = [traj_id for traj_id in range(args.num_trajectories) if traj_id not in completed_traj_ids]
    if not pending_traj_ids:
        return [traj for traj in trajectories if traj is not None]

    def worker(traj_id: int) -> Dict:
        response_data = call_openrouter(
            client=client,
            args=args,
            prompt_text=prompt_item["prompt_text"],
            seed=args.seed + prompt_item["prompt_id"] * 1000 + traj_id,
        )
        content_text = response_data["content"]
        reasoning_text = response_data["reasoning"]
        full_text = combine_reasoning_and_content(reasoning_text, content_text)
        return {
            "traj_id": traj_id,
            "full_text": full_text,
            "reasoning_text": reasoning_text,
            "prefix_text": make_prefix_text(reasoning_text, args.prefix_token_index),
            "prefix_hidden_state": None,
            "reward": compute_reward(full_text, prompt_item["problem_id"], args.testcase_root),
        }

    with ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
        future_map = {executor.submit(worker, traj_id): traj_id for traj_id in pending_traj_ids}
        progress_desc = f"prompt {prompt_item['prompt_id']} trajectories"
        for future in tqdm(
            as_completed(future_map),
            total=len(pending_traj_ids),
            desc=progress_desc,
            leave=False,
        ):
            result = future.result()
            trajectories[result["traj_id"]] = result
            if existing_prompt_result is not None:
                existing_prompt_result["trajectories"] = [traj for traj in trajectories if traj is not None]
            if save_callback is not None:
                save_callback()

    return [traj for traj in trajectories if traj is not None]


def main() -> None:
    args = parse_args()
    apply_smoke_test_overrides(args)
    set_seed(args.seed)

    if not args.api_key:
        raise ValueError("Missing OpenRouter API key. Pass --api-key or set OPENROUTER_API_KEY.")

    headers = {}
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        default_headers=headers or None,
    )

    if args.probe_reasoning:
        probe = call_openrouter(
            client=client,
            args=args,
            prompt_text="Write a C++ program that reads two integers and prints their sum.",
            seed=args.seed,
        )
        print("HAS_REASONING=", bool(probe["reasoning"].strip()))
        print("REASONING_PREVIEW=", probe["reasoning"][:500])
        print("CONTENT_PREVIEW=", probe["content"][:500])
        return

    dataset = load_programming_dataset(args)
    sample_size = min(args.num_prompts, len(dataset))
    chosen_indices = random.sample(range(len(dataset)), sample_size)

    existing_results = load_existing_results(args.output_path) if args.resume else []
    existing_index = build_results_index(existing_results)
    results: List[Dict] = []
    for prompt_order, dataset_idx in tqdm(
        list(enumerate(chosen_indices)),
        total=len(chosen_indices),
        desc="prompts",
    ):
        sample = dataset[dataset_idx]
        prompt_text = str(sample[args.question_column])
        problem_id = str(sample[args.id_column])
        existing_prompt_result = existing_index.get(int(dataset_idx))
        if existing_prompt_result is None:
            existing_prompt_result = {
                "prompt_id": prompt_order,
                "prompt_text": prompt_text,
                "dataset_idx": int(dataset_idx),
                "problem_id": problem_id,
                "gold_answer": None,
                "hidden_state_source": "unavailable_from_openrouter_chat_completions_api",
                "trajectories": [],
            }
            existing_index[int(dataset_idx)] = existing_prompt_result
        else:
            existing_prompt_result["prompt_id"] = prompt_order
            existing_prompt_result["prompt_text"] = prompt_text
            existing_prompt_result["problem_id"] = problem_id
            existing_prompt_result["gold_answer"] = None

        def save_partial_results() -> None:
            ordered_results = []
            for ordered_prompt_id, ordered_dataset_idx in enumerate(chosen_indices):
                item = existing_index.get(int(ordered_dataset_idx))
                if item is not None:
                    item["prompt_id"] = ordered_prompt_id
                    ordered_results.append(item)
            atomic_pickle_dump(args.output_path, ordered_results)

        trajectories = generate_trajectories_for_prompt(
            client=client,
            args=args,
            prompt_item={
                "prompt_id": prompt_order,
                "prompt_text": prompt_text,
                "problem_id": problem_id,
            },
            existing_prompt_result=existing_prompt_result,
            save_callback=save_partial_results,
        )
        existing_prompt_result["trajectories"] = trajectories
        results.append(existing_prompt_result)
        save_partial_results()

    atomic_pickle_dump(args.output_path, results)


if __name__ == "__main__":
    main()
