#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from code_evals import extract_last_cpp_block


METHOD_TO_KEY = {
    "hidden": "cluster_id_hidden",
    "embedding": "cluster_id_embedding",
    "knn": "cluster_id_knn",
    "random": "cluster_id_random",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show clustering results for one prompt.")
    parser.add_argument("--input-path", default="clustered_data.pkl")
    parser.add_argument("--prompt-id", type=int, default=None)
    parser.add_argument("--method", choices=sorted(METHOD_TO_KEY.keys()), default="embedding")
    parser.add_argument("--max-text-chars", type=int, default=240)
    parser.add_argument("--max-items-per-cluster", type=int, default=10)
    parser.add_argument("--show-code", action="store_true")
    parser.add_argument("--max-code-lines", type=int, default=40)
    parser.add_argument("--sort-by", choices=["cluster", "reward"], default="cluster")
    parser.add_argument("--min-accuracy", type=float, default=0.2)
    parser.add_argument("--max-accuracy", type=float, default=0.8)
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include prompts with zero reward variance, e.g. all-wrong or all-correct cases.",
    )
    parser.add_argument(
        "--select",
        choices=["manual", "highest-accuracy", "best-homogeneity", "best-mid-accuracy"],
        default="manual",
    )
    return parser.parse_args()


def shorten(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def shorten_code(text: str, max_lines: int) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines).strip()
    kept = "\n".join(lines[:max_lines]).strip()
    return f"{kept}\n... ({len(lines) - max_lines} more lines)"


def find_prompt(data: List[Dict], prompt_id: int) -> Dict:
    for item in data:
        if item["prompt_id"] == prompt_id:
            return item
    raise ValueError(f"prompt_id={prompt_id} not found.")


def find_highest_accuracy_prompt(data: List[Dict]) -> Dict:
    best_item = None
    best_accuracy = -1.0
    best_reward_sum = -1
    for item in data:
        trajectories = item.get("trajectories", [])
        if not trajectories:
            continue
        rewards = [traj.get("reward", 0) for traj in trajectories]
        accuracy = float(np.mean(rewards))
        reward_sum = int(sum(rewards))
        if accuracy > best_accuracy or (accuracy == best_accuracy and reward_sum > best_reward_sum):
            best_item = item
            best_accuracy = accuracy
            best_reward_sum = reward_sum
    if best_item is None:
        raise ValueError("No prompt with trajectories found.")
    return best_item


def prompt_accuracy(prompt_item: Dict) -> float | None:
    trajectories = prompt_item.get("trajectories", [])
    if not trajectories:
        return None
    rewards = [traj.get("reward", 0) for traj in trajectories]
    return float(np.mean(rewards))


def weighted_intra_cluster_variance(rewards: np.ndarray, labels: np.ndarray) -> float:
    total = len(rewards)
    weighted_sum = 0.0
    for cluster_id in np.unique(labels):
        cluster_rewards = rewards[labels == cluster_id]
        cluster_var = 0.0 if len(cluster_rewards) <= 1 else float(np.var(cluster_rewards))
        weighted_sum += len(cluster_rewards) * cluster_var
    return weighted_sum / total if total > 0 else 0.0


def per_prompt_variance_ratio(prompt_item: Dict, label_key: str) -> float | None:
    trajectories = prompt_item.get("trajectories", [])
    if not trajectories or any(traj.get(label_key) is None for traj in trajectories):
        return None
    rewards = np.array([traj["reward"] for traj in trajectories], dtype=float)
    labels = np.array([traj[label_key] for traj in trajectories], dtype=int)
    var_prompt = float(np.var(rewards))
    var_intra = weighted_intra_cluster_variance(rewards, labels)
    if var_prompt == 0:
        return 0.0 if var_intra == 0 else None
    return var_intra / var_prompt


def is_degenerate_prompt(prompt_item: Dict) -> bool:
    trajectories = prompt_item.get("trajectories", [])
    if not trajectories:
        return True
    rewards = np.array([traj["reward"] for traj in trajectories], dtype=float)
    return float(np.var(rewards)) == 0.0


def find_best_homogeneity_prompt(data: List[Dict], label_key: str, include_degenerate: bool) -> Dict:
    best_item = None
    best_ratio = None
    for item in data:
        if not include_degenerate and is_degenerate_prompt(item):
            continue
        ratio = per_prompt_variance_ratio(item, label_key)
        if ratio is None:
            continue
        if best_ratio is None or ratio < best_ratio:
            best_item = item
            best_ratio = ratio
    if best_item is None:
        raise ValueError(f"No prompt with valid labels found for {label_key}.")
    return best_item


def find_best_mid_accuracy_prompt(
    data: List[Dict],
    label_key: str,
    include_degenerate: bool,
    min_accuracy: float,
    max_accuracy: float,
) -> Dict:
    best_item = None
    best_ratio = None
    for item in data:
        if not include_degenerate and is_degenerate_prompt(item):
            continue
        accuracy = prompt_accuracy(item)
        if accuracy is None or accuracy < min_accuracy or accuracy > max_accuracy:
            continue
        ratio = per_prompt_variance_ratio(item, label_key)
        if ratio is None:
            continue
        if best_ratio is None or ratio < best_ratio:
            best_item = item
            best_ratio = ratio
    if best_item is None:
        raise ValueError(
            f"No prompt found in accuracy range [{min_accuracy}, {max_accuracy}] with valid labels for {label_key}."
        )
    return best_item


def cluster_summary(trajectories: List[Dict], label_key: str) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for traj in trajectories:
        label = traj.get(label_key)
        if label is None:
            continue
        grouped[int(label)].append(traj)
    return dict(grouped)


def print_cluster(
    cluster_id: int,
    items: List[Dict],
    max_text_chars: int,
    max_items: int,
    show_code: bool,
    max_code_lines: int,
) -> None:
    rewards = np.array([item["reward"] for item in items], dtype=float)
    mean_reward = float(np.mean(rewards)) if len(rewards) else float("nan")
    print(f"\n[Cluster {cluster_id}] size={len(items)} mean_reward={mean_reward:.3f}")
    for traj in items[:max_items]:
        reasoning = traj.get("reasoning_text") or traj.get("prefix_text") or traj.get("full_text", "")
        strategy = traj.get("strategy_text", "")
        final_text = traj.get("full_text", "")
        print(
            f"  traj_id={traj['traj_id']:>2} reward={traj['reward']} "
            f"prefix={shorten(reasoning, max_text_chars)}"
        )
        if strategy:
            print(f"    strategy={shorten(strategy, max_text_chars)}")
        print(f"    full={shorten(final_text, max_text_chars)}")
        if show_code:
            code_text = extract_last_cpp_block(final_text)
            if code_text:
                print("    code:")
                for line in shorten_code(code_text, max_code_lines).splitlines():
                    print(f"      {line}")
    remaining = len(items) - max_items
    if remaining > 0:
        print(f"  ... {remaining} more trajectories omitted")


def main() -> None:
    args = parse_args()
    label_key = METHOD_TO_KEY[args.method]

    with open(args.input_path, "rb") as f:
        data = pickle.load(f)

    if args.select == "highest-accuracy":
        prompt_item = find_highest_accuracy_prompt(data)
    elif args.select == "best-homogeneity":
        prompt_item = find_best_homogeneity_prompt(data, label_key, args.include_degenerate)
    elif args.select == "best-mid-accuracy":
        prompt_item = find_best_mid_accuracy_prompt(
            data,
            label_key,
            args.include_degenerate,
            args.min_accuracy,
            args.max_accuracy,
        )
    else:
        if args.prompt_id is None:
            raise ValueError("Please pass --prompt-id or use --select highest-accuracy.")
        prompt_item = find_prompt(data, args.prompt_id)
    trajectories = prompt_item["trajectories"]
    grouped = cluster_summary(trajectories, label_key)

    if not grouped:
        raise ValueError(f"No clustering labels found for method={args.method} on prompt_id={args.prompt_id}.")

    print(f"prompt_id={prompt_item['prompt_id']}")
    print(f"dataset_idx={prompt_item.get('dataset_idx')}")
    print(f"method={args.method}")
    print(f"gold_answer={prompt_item.get('gold_answer')}")
    print(f"prompt={shorten(prompt_item.get('prompt_text', ''), args.max_text_chars)}")
    accuracy = prompt_accuracy(prompt_item)
    if accuracy is not None:
        print(f"accuracy={accuracy:.3f}")
    ratio = per_prompt_variance_ratio(prompt_item, label_key)
    if ratio is not None:
        print(f"var_intra_over_var_prompt={ratio:.4f}")
    print(f"num_clusters={len(grouped)}")

    cluster_items = list(grouped.items())
    if args.sort_by == "reward":
        cluster_items.sort(key=lambda x: np.mean([item["reward"] for item in x[1]]), reverse=True)
    else:
        cluster_items.sort(key=lambda x: x[0])

    for cluster_id, items in cluster_items:
        items = sorted(items, key=lambda x: (x["reward"], x["traj_id"]), reverse=True)
        print_cluster(
            cluster_id,
            items,
            args.max_text_chars,
            args.max_items_per_cluster,
            args.show_code,
            args.max_code_lines,
        )


if __name__ == "__main__":
    main()
