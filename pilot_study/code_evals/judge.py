import os
import re
import subprocess
import uuid
from pathlib import Path


DEFAULT_TESTCASE_ROOT = Path(
    os.environ.get("CODE_EVAL_TESTCASE_ROOT", "/data/leesin5202/version_1/LCB_eval/testcases")
)
JUDGER_PATH = Path(__file__).with_name("judger.sh")


def extract_last_cpp_block(text: str) -> str:
    matches = re.findall(r"```(?:cpp|c\+\+)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()


def judge(displayed_id: int | str, code_str: str, testcase_root: str | os.PathLike | None = None) -> int:
    no = str(uuid.uuid4())
    testcase_base = Path(testcase_root) if testcase_root is not None else DEFAULT_TESTCASE_ROOT
    path = testcase_base / str(displayed_id)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing testcase directory for {displayed_id}: {path}")

    code_str = extract_last_cpp_block(code_str)

    source_path = path / f"temp_{no}.cpp"
    with open(source_path, "w") as f:
        f.write(code_str)

    try:
        result = subprocess.run(
            ["bash", str(JUDGER_PATH), str(path), no],
            text=True,
            capture_output=True,
        )
    finally:
        if source_path.exists():
            source_path.unlink()

    if result.returncode != 0:
        raise ValueError(f"Encountered an error when judging {displayed_id}: {result.stderr}")

    return int(result.stdout.strip())


if __name__ == '__main__':
    code = r'''
    #include<bits/stdc++.h>

using namespace std;

int main() {
    int n;
    string s;
    cin >> n >> s;

    int t = 0, a = 0;
    for (int i = 0; i < n; i++) {
        if (s[i] == 'T') ++t;
        else ++a;
    }

    if (t > a) cout << 'T' << endl;
    else if (t < a) cout << 'A' << endl;
    else cout << char('T' + 'A' - s.back()) << endl;
}

'''

    print(judge('atcoder/abc301_a', code))
