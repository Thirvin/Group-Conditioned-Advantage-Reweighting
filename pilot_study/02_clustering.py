#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from typing import List

import numpy as np
from openai import APIError, APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from scipy.sparse.csgraph import connected_components

STRATEGY_SYSTEM_PROMPT = """You convert a programming-problem reasoning prefix into a compact representation of the model's current thinking logic for clustering.

Requirements:
- Focus on the reasoning process already present in the prefix: what the model is trying to establish, what assumptions it is making, what subproblem it is reducing to, what conditions it checks next, and how it chooses the next step.
- Preserve decision structure, causal order, comparisons, invariants, state transitions, and edge-case considerations when they appear.
- Preserve code-level relations only when they reflect the reasoning logic, such as loop intent, update rules, indexing logic, or dependency order.
- Do not rewrite the prefix into a standalone solution template or a generic algorithm label.
- Output short structured plain text, not JSON.
- Do not solve the problem further; only summarize the thinking logic already present in the prefix.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster trajectories with hidden states and embeddings.")
    parser.add_argument("--input-path", default="rollouts_data.pkl")
    parser.add_argument("--output-path", default="clustered_data.pkl")
    parser.add_argument("--prefix-token-index", type=int, default=None)
    parser.add_argument("--prefix-tokenizer-name", default="google/gemma-3-27b-it")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--distance-quantile", type=float, default=0.35)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--strategy-source", choices=["llm", "raw"], default="llm")
    parser.add_argument("--strategy-cache-path", default="strategy_cache.json")
    parser.add_argument("--strategy-model", default="google/gemma-4-31b")
    parser.add_argument("--strategy-api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--strategy-base-url", default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--strategy-http-referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"))
    parser.add_argument("--strategy-x-title", default=os.environ.get("OPENROUTER_X_TITLE", "pilot-study-clustering"))
    parser.add_argument("--strategy-max-output-tokens", type=int, default=192)
    parser.add_argument("--strategy-temperature", type=float, default=0.0)
    parser.add_argument("--strategy-max-retries", type=int, default=8)
    parser.add_argument("--strategy-retry-backoff", type=float, default=2.0)
    return parser.parse_args()


def adaptive_distance_threshold(features: np.ndarray, quantile: float) -> float:
    if len(features) <= 1:
        return 0.0
    distances = pairwise_distances(features, metric="euclidean")
    upper_tri = distances[np.triu_indices_from(distances, k=1)]
    if upper_tri.size == 0:
        return 0.0
    threshold = float(np.quantile(upper_tri, quantile))
    return max(threshold, 1e-8)


def cluster_with_agglomerative(features: np.ndarray, quantile: float) -> np.ndarray:
    if len(features) == 1:
        return np.array([0], dtype=int)
    threshold = adaptive_distance_threshold(features, quantile)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
    )
    return model.fit_predict(features)


def cluster_with_knn_graph(features: np.ndarray, k: int) -> np.ndarray:
    if len(features) == 1:
        return np.array([0], dtype=int)
    effective_k = max(1, min(k, len(features) - 1))
    graph = kneighbors_graph(
        features,
        n_neighbors=effective_k,
        mode="connectivity",
        include_self=False,
    )
    undirected_graph = graph.maximum(graph.T)
    _, labels = connected_components(undirected_graph, directed=False)
    return labels.astype(int)


def truncate_prefix_text(text: str, prefix_token_index: int | None, tokenizer) -> str:
    if prefix_token_index is None or tokenizer is None:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    prefix_token_ids = token_ids[:prefix_token_index]
    if not prefix_token_ids:
        return ""
    return tokenizer.decode(prefix_token_ids, skip_special_tokens=True)


def get_prefix_texts(trajectories: List[dict], prefix_token_index: int | None, tokenizer) -> List[str]:
    texts = []
    for traj in trajectories:
        base_text = (
            traj.get("reasoning_text")
            or traj.get("prefix_text")
            or traj.get("prefix_text_64")
            or traj.get("full_text", "")
        )
        texts.append(truncate_prefix_text(base_text, prefix_token_index, tokenizer))
    return texts


def load_prefix_tokenizer(model_name: str):
    candidates = [model_name]

    # OpenRouter/provider-style ids are not always valid Hugging Face tokenizer repos.
    normalized = model_name.strip()
    if "/" in normalized:
        provider, repo = normalized.split("/", 1)
        if provider in {"google", "openai", "anthropic", "meta-llama", "mistralai"}:
            candidates.append(repo)

    lowered = normalized.lower()
    if "gemma" in lowered:
        candidates.extend(
            [
                "google/gemma-2-9b-it",
                "google/gemma-2-2b-it",
            ]
        )

    seen = set()
    deduped_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    last_error = None
    for candidate in deduped_candidates:
        for use_fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    use_fast=use_fast,
                )
            except Exception as error:
                last_error = error

    raise RuntimeError(
        f"Failed to load a tokenizer for prefix truncation. Tried: {deduped_candidates}. "
        f"Last error: {last_error}"
    )


def random_labels_from_reference(reference_labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_labels, counts = np.unique(reference_labels, return_counts=True)
    shuffled_indices = rng.permutation(len(reference_labels))
    random_labels = np.empty(len(reference_labels), dtype=int)
    start = 0
    for new_label, count in enumerate(counts):
        selected = shuffled_indices[start : start + count]
        random_labels[selected] = new_label
        start += count
    return random_labels


def atomic_json_dump(path: str, obj: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path) or ".", delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_strategy_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strategy_cache_key(model_name: str, prefix_text: str) -> str:
    payload = f"{model_name}\n{prefix_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_strategy_client(args: argparse.Namespace):
    if args.strategy_source != "llm":
        return None
    if not args.strategy_api_key:
        raise ValueError("Missing OpenRouter API key for strategy extraction. Set OPENROUTER_API_KEY or pass --strategy-api-key.")
    headers = {}
    if args.strategy_http_referer:
        headers["HTTP-Referer"] = args.strategy_http_referer
    if args.strategy_x_title:
        headers["X-Title"] = args.strategy_x_title
    return OpenAI(
        api_key=args.strategy_api_key,
        base_url=args.strategy_base_url,
        default_headers=headers or None,
    )


def extract_retry_delay_seconds(error: Exception) -> float | None:
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


def extract_strategy_with_llm(client: OpenAI, args: argparse.Namespace, prefix_text: str) -> str:
    last_error = None
    for attempt in range(args.strategy_max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.strategy_model,
                messages=[
                    {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
                    {"role": "user", "content": prefix_text},
                ],
                temperature=args.strategy_temperature,
                max_tokens=args.strategy_max_output_tokens,
            )
            content = response.choices[0].message.content
            return (content if isinstance(content, str) else str(content)).strip()
        except RateLimitError as error:
            last_error = error
            if attempt >= args.strategy_max_retries:
                raise
            time.sleep(extract_retry_delay_seconds(error) or args.strategy_retry_backoff * (attempt + 1))
        except (APIError, APIConnectionError, APITimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= args.strategy_max_retries:
                raise
            time.sleep(args.strategy_retry_backoff * (attempt + 1))
    raise RuntimeError(f"Strategy extraction failed after retries: {last_error}")


def get_strategy_texts(prefix_texts: List[str], args: argparse.Namespace, client, cache: dict) -> List[str]:
    strategy_texts: List[str] = []
    updated = False
    progress = tqdm(prefix_texts, desc="extract strategies", leave=False)
    for prefix_text in progress:
        if args.strategy_source == "raw":
            strategy_texts.append(prefix_text)
            continue
        cache_key = strategy_cache_key(args.strategy_model, prefix_text)
        cached = cache.get(cache_key)
        if cached is None:
            cached = extract_strategy_with_llm(client, args, prefix_text)
            cache[cache_key] = cached
            updated = True
            atomic_json_dump(args.strategy_cache_path, cache)
        strategy_texts.append(cached)
    if updated:
        atomic_json_dump(args.strategy_cache_path, cache)
    return strategy_texts


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    embedding_model = SentenceTransformer(args.embedding_model, device='cpu')
    strategy_client = build_strategy_client(args)
    strategy_cache = load_strategy_cache(args.strategy_cache_path)
    prefix_tokenizer = None
    if args.prefix_token_index is not None:
        prefix_tokenizer = load_prefix_tokenizer(args.prefix_tokenizer_name)

    with open(args.input_path, "rb") as f:
        data = pickle.load(f)

    for prompt_item in tqdm(data, desc="clustering prompts"):
        trajectories = prompt_item["trajectories"]
        hidden_vectors = [traj.get("prefix_hidden_state") for traj in trajectories]
        hidden_available = all(vec is not None for vec in hidden_vectors)
        if hidden_available:
            hidden_matrix = np.stack(hidden_vectors, axis=0)
            pca_dim = min(args.pca_dim, hidden_matrix.shape[0], hidden_matrix.shape[1])
            if pca_dim >= 1:
                reduced_hidden = PCA(n_components=pca_dim, random_state=0).fit_transform(hidden_matrix)
            else:
                reduced_hidden = hidden_matrix
            hidden_labels = cluster_with_agglomerative(reduced_hidden, args.distance_quantile)
        else:
            hidden_labels = None

        prefix_texts = get_prefix_texts(trajectories, args.prefix_token_index, prefix_tokenizer)
        strategy_texts = get_strategy_texts(prefix_texts, args, strategy_client, strategy_cache)
        if any(text.strip() for text in strategy_texts):
            embedding_matrix = embedding_model.encode(
                strategy_texts,
                batch_size=args.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            embedding_matrix = np.zeros((len(prefix_texts), 1), dtype=float)
        embedding_labels = cluster_with_agglomerative(embedding_matrix, args.distance_quantile)
        knn_labels = cluster_with_knn_graph(embedding_matrix, args.knn_k)
        random_labels = random_labels_from_reference(embedding_labels, rng)

        for idx, (traj, embedding_label) in enumerate(zip(trajectories, embedding_labels)):
            traj["cluster_id_hidden"] = None if hidden_labels is None else int(hidden_labels[idx])
            traj["cluster_id_embedding"] = int(embedding_label)
            traj["cluster_id_knn"] = int(knn_labels[idx])
            traj["cluster_id_random"] = int(random_labels[idx])
            traj["prefix_text"] = prefix_texts[idx]
            traj["strategy_text"] = strategy_texts[idx]

    with open(args.output_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
