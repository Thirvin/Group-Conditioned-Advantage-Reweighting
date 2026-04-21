#!/usr/bin/env python3
import argparse
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cluster homogeneity metrics.")
    parser.add_argument("--input-path", default="clustered_data.pkl")
    return parser.parse_args()


def weighted_intra_cluster_variance(rewards: np.ndarray, labels: np.ndarray) -> float:
    total = len(rewards)
    weighted_sum = 0.0
    for cluster_id in np.unique(labels):
        cluster_rewards = rewards[labels == cluster_id]
        if len(cluster_rewards) <= 1:
            cluster_var = 0.0
        else:
            cluster_var = float(np.var(cluster_rewards))
        weighted_sum += len(cluster_rewards) * cluster_var
    return weighted_sum / total if total > 0 else 0.0


def lucky_guess_rate(rewards: np.ndarray, labels: np.ndarray) -> float:
    positive_indices = np.where(rewards == 1)[0]
    if len(positive_indices) == 0:
        return 0.0

    lucky_count = 0
    for idx in positive_indices:
        cluster_mask = labels == labels[idx]
        cluster_indices = np.where(cluster_mask)[0]
        peer_indices = cluster_indices[cluster_indices != idx]
        if len(peer_indices) == 0:
            loo_mean = 0.0
        else:
            loo_mean = float(np.mean(rewards[peer_indices]))
        if loo_mean < 0.2:
            lucky_count += 1
    return lucky_count / len(positive_indices)


def singleton_frequency(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) == 0:
        return 0.0
    return float(np.mean(counts == 1))


def evaluate_method(data: List[dict], label_key: str) -> Dict[str, float]:
    ratio_values = []
    lucky_values = []
    k_values = []
    singleton_values = []

    for prompt_item in tqdm(data, desc=f"eval {label_key}", leave=False):
        trajectories = prompt_item["trajectories"]
        if not trajectories or any(traj.get(label_key) is None for traj in trajectories):
            continue
        rewards = np.array([traj["reward"] for traj in trajectories], dtype=float)
        labels = np.array([traj[label_key] for traj in trajectories], dtype=int)

        var_prompt = float(np.var(rewards))
        var_intra = weighted_intra_cluster_variance(rewards, labels)
        if var_prompt == 0:
            ratio = 0.0 if var_intra == 0 else np.nan
        else:
            ratio = var_intra / var_prompt
        ratio_values.append(ratio)

        lucky_values.append(lucky_guess_rate(rewards, labels))
        k_values.append(len(np.unique(labels)))
        singleton_values.append(singleton_frequency(labels))

    ratio_array = np.array(ratio_values, dtype=float)
    return {
        "Var_intra / Var_prompt": float(np.nanmean(ratio_array)) if len(ratio_array) else np.nan,
        "Lucky Guess Rate": float(np.mean(lucky_values)) if lucky_values else np.nan,
        "Avg #Clusters (K)": float(np.mean(k_values)) if k_values else np.nan,
        "Singleton Cluster Freq": float(np.mean(singleton_values)) if singleton_values else np.nan,
    }


def main() -> None:
    args = parse_args()
    with open(args.input_path, "rb") as f:
        data = pickle.load(f)

    rows = {
        "hidden": evaluate_method(data, "cluster_id_hidden"),
        "embedding": evaluate_method(data, "cluster_id_embedding"),
        "knn": evaluate_method(data, "cluster_id_knn"),
        "random": evaluate_method(data, "cluster_id_random"),
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "method"
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df)


if __name__ == "__main__":
    main()
