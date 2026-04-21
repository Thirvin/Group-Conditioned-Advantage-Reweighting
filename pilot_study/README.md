我正在進行一項 LLM 強化學習的研究。目前我們先把實驗目標從數學題改成程式問題，想觀察模型在解 code / debugging / algorithm 類題目時，是否更容易形成可分群的策略模式。

目前這個 repo 已經有一條可跑的 rollout generation + clustering + evaluation pipeline，現在預設已切到「程式問題」實驗；數學題只視為舊實驗背景。

## 使用 uv 管理環境

本專案已可直接用 `uv` 管理 Python 環境與依賴。

```bash
uv sync
```

若你要執行三個步驟，建議直接用：

```bash
uv run python 01_generate_rollouts.py
uv run python 01_generate_rollouts_api.py
uv run python 02_clustering.py
uv run python 03_metrics_eval.py
```

也可以直接用 `Makefile` 或 shell script 跑完整 pipeline：

```bash
make smoke-api
make smoke-local

MODE=api SIZE=smoke ./run_pipeline.sh
MODE=local SIZE=full ./run_pipeline.sh
```

也可以帶參數，例如：

```bash
uv run python 01_generate_rollouts.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --hidden-batch-size 8 \
  --gpu-memory-utilization 0.85
```

`01_generate_rollouts.py` 現在使用 `vLLM` 來加速 rollout generation，再用 `transformers` 對每條 trajectory 的前綴做一次 forward pass，提取第 `N` 個 generated token 對應的最後一層 hidden state。

如果你只想先快速驗證 pipeline，可用 smoke test：

```bash
uv run python 01_generate_rollouts.py --smoke-test
```

這會自動縮成 2 題、每題 4 條 trajectories，並降低 hidden-state 抽取與生成長度，方便先檢查流程是否正常。

## 用 OpenRouter API 生成 Rollouts

如果本機 VRAM 不夠，可改用 OpenRouter 的 OpenAI-compatible API 生成 rollout。OpenRouter 官方文件的 chat completions endpoint 是 `https://openrouter.ai/api/v1/chat/completions`，可選擇 Gemma 系列模型並帶上 `HTTP-Referer` / `X-Title` header。

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_HTTP_REFERER="https://your-site.example"
export OPENROUTER_X_TITLE="pilot-study"

uv run python 01_generate_rollouts_api.py \
  --model-name google/gemma-3-27b-it \
  --smoke-test
```

如果遇到 `429` / rate limit exceeded，先把 `--parallel-requests` 降到 `1` 或 `2`；目前腳本也會自動依照錯誤訊息中的重試時間做退避重試。

API 版會保留 `full_text`、`prefix_text`、`reward`，但 OpenRouter `chat.completions` API 不會回傳 hidden states，所以 `prefix_hidden_state` 會是 `None`。
這種情況下：

- `02_clustering.py` 仍可執行 `embedding` baseline
- `hidden` clustering 會被略過
- `03_metrics_eval.py` 會對 `hidden` 指標輸出 `NaN`

API 版生成腳本現在也支援 checkpoint / resume：

- 每完成一條 trajectory 就會把部分結果寫回 `rollouts_data.pkl`
- 重新執行同一組參數時，會先讀既有輸出並跳過已完成的 trajectories
- 如需強制從頭跑，可加 `--no-resume`

如果需要重建 lock file：

```bash
uv lock
```

## `code_evals`：程式題資料與 reward evaluator

repo 內目前已放了一套偏競程風格的 code evaluation 原型，位於 `code_evals/`，可作為把 reward 從數學題改成「程式題測資通過率」的起點。

### 內容概覽

- `code_evals/LCB_dataset.csv`
  - 目前的程式題 prompt 資料。
  - 欄位至少包含：
    - `no`：題目 ID，例如 `atcoder/abc370_a`
    - `text`：完整題述
- `code_evals/judge.py`
  - Python 入口函式 `judge(displayed_id, code_str)`。
  - 會把模型生成的 C++ 程式碼寫成暫存檔，呼叫 shell judger，最後回傳整數狀態碼。
- `code_evals/judger.sh`
  - 實際執行編譯與測資比對。
  - 目前使用 `g++ -O2 -std=gnu++2a` 編譯，代表 evaluator 目前是以 C++ 解答為前提。

### reward 定義

`judge.py` / `judger.sh` 目前提供的是非常直接的 binary reward：

- `1`：compile 成功，且全部 testcase 輸出都正確
- `0`：compile 成功，但至少一筆 testcase 答錯
- `-1`：compile 失敗，或 testcase 檔案不完整 / judging 過程異常

`judger.sh` 也對每次執行加了基本限制：

- CPU time limit：`ulimit -t 2`
- virtual memory limit：`ulimit -v 1048576`

### 目前 judge 的工作方式

給定題目 ID 與模型輸出的 C++ 程式碼後：

1. `judge.py` 會先移除 markdown code fence，例如 ````cpp` 與 ```。
   若回覆裡有多個 fenced code block，會抓最後一個 ````cpp ... ```` 片段當作最終答案。
2. 它會把程式碼寫到對應 testcase 目錄下的暫存檔 `temp_<uuid>.cpp`。
3. `judger.sh` 會在該題 testcase 目錄中編譯程式。
4. 編譯成功後，會逐一讀取 `*.in` / `*.out` 測資做比對。
5. 全部通過回傳 `1`；有任何 testcase 不符回傳 `0`；編譯失敗回傳 `-1`。

### 目前整合狀態與限制

這套 `code_evals` 原型現在已接到 `01_generate_rollouts.py` 與 `01_generate_rollouts_api.py`。主流程腳本目前預設會：

- 讀取 `code_evals/LCB_dataset.csv` 當作 prompt source
- 要求模型把最終 C++ 解答放在最後一個 ```cpp ... ``` code block
- 用 `code_evals.judge.judge(...)` 對該 code block 做 testcase-based reward 計算

目前 rollout generation 端已經切到程式題；若之後還要往前走，下一步會是：

- 擴充到不只 C++ 的 evaluator
- 把 testcase root 與資料路徑進一步做成更可攜的設定
- 視需要把 compile error / wrong answer / runtime error 拆成更細的 reward signal

目前還有一個實作層面的限制要注意：

- testcase root 預設仍是 `/data/leesin5202/version_1/LCB_eval/testcases`，但現在可用 `CODE_EVAL_TESTCASE_ROOT` 或 rollout 腳本的 `--testcase-root` 覆寫

## 目前實驗目標

1. 蒐集一批「程式問題」prompt 的多條 rollout。
2. 從每條 rollout 的早期推理 prefix 中提取可比較的策略表徵。
3. 針對同一題下的多條 trajectories 做策略分群。
4. 驗證 cluster 是否比 prompt-level reward 更 homogeneous。

## 目前 pipeline 在做什麼

### `01_generate_rollouts.py`

- 以本地模型生成多條 rollout。
- 若資源允許，也可抽取 prefix hidden states。
- 適合之後要做 hidden-state clustering 的情境。

### `01_generate_rollouts_api.py`

- 以 OpenRouter API 生成多條 rollout。
- 目前支援 reasoning 回傳、checkpoint / resume、prefix 儲存。
- 適合先快速蒐集大量程式問題 rollout。

### `02_clustering.py`

- 目前支援三類分群：
  - hidden state clustering
  - embedding clustering
  - kNN graph clustering
- 另外保留 matched random baseline。
- 目前 clustering 可先對 prefix 做 tokenizer-based truncation，再用另一個 LLM 提取 strategy text，最後基於 strategy text 做 embedding / kNN 分群。

### `03_metrics_eval.py`

- 計算：
  - `Var_intra / Var_prompt`
  - `Lucky Guess Rate`
  - `Avg #Clusters (K)`
  - `Singleton Cluster Freq`

### `04_show_clustering.py`

- 展示單一 prompt 的 clustering 結果。
- 可自動選擇：
  - highest-accuracy case
  - best-homogeneity case
  - best-mid-accuracy case
- 可檢視原始 reasoning、strategy text、完整輸出。

## 下一步方向

接下來建議把資料集與 reward 定義正式切到程式問題，例如：

- 題目型態：bug fixing、程式理解、演算法設計、code completion、unit-test repair
- reward：通過測試、compile success、execution success、hidden test pass rate、AST-level correctness
- 若沿用目前 `code_evals` 原型，最直接的第一步就是先用 C++ testcase pass/fail 當 binary reward
- prefix：優先觀察 reasoning / strategy，而不是最終答案文字

目前 repo 的技術骨架已可重用；真正需要替換的是：

- prompt / dataset 來源
- reward 計算方式
- 若做 API 實驗，最好把 reasoning-prefix strategy extraction 保留下來
