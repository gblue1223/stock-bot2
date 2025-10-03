#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
자동 재학습 스크립트
- 기본 학습 CLI를 실행합니다.
- 학습 스크립트가 자동 진단 코드(2)로 종료되면, 최근 로그를 요약해 ChatGPT(OpenAI API)에 질의하여
  안전한 범위의 하이퍼파라미터(임계값/호라이즌/LR/WD/배치) 조정을 제안받습니다.
- 제안 중 허용된 항목만 적용하여 CLI를 수정하고, 재학습을 재시도합니다(최대 N회).

사전 준비
- 환경변수 OPENAI_API_KEY 를 설정해 주세요. (절대 키를 코드/레포에 저장하지 마세요.)
- requirements: requests (이미 있음)

사용 예
  python scripts/auto_retrain_supervised_ml.py

주의
- 현재 학습 코드는 회귀(mse/huber)와 분류(ce/focal)를 지원합니다.
- 안전한 변경 우선: direction3-threshold, horizon, lr, weight-decay, batch-size, label-smoothing, focal-gamma, loss, seq-len
"""
import os
import sys
import json
import shlex
import time
import subprocess
import textwrap
import requests

from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv

# dotenv
try:
    _DOTENV_PATH = find_dotenv()
    if _DOTENV_PATH:
        load_dotenv(_DOTENV_PATH, override=False)
        print(f"[정보] .env 로드: {_DOTENV_PATH}")
    else:
        # 기본적으로 현재 작업 디렉터리에서 .env를 탐색하지 못한 경우 무시
        pass
except Exception:
    # python-dotenv 미설치 시 무시 (requirements에 추가 권장)
    pass

# ---------------- 설정 ----------------
# 메모리 파일 경로 (이 스크립트와 같은 디렉터리 아래에 생성)
MEMORY_PATH = os.path.join(os.path.dirname(__file__), 
                           ".auto_retrain_supervised_ml.memory.json")
# 기본 학습 CLI (사용자 예시 기반)
BASE_CMD: List[str] = shlex.split(
    r'''./.venv64/Scripts/python -m ai_trader.ml.train_supervised
  --db "C:\Users\user\Workspace\datasets@20251002\datasets_norm_all.duckdb"
  --table datasets
  --out models/supervised
  --seq-len 60
  --horizon 20
  --chunk-size 50000
  --device cuda
  --batch-size 128
  --epochs 10
  --lr 1e-4
  --weight-decay 1e-4
  --progress-every 10
  --ckpt-every-chunks 50
  --resume-from models/supervised/checkpoints
  --target-col 현재가
 
  --aux-task direction3
  --loss focal
  --focal-gamma 2.0
  --use-weighted-sampler
  --direction3-threshold 0.015
 
  --auto-diagnosis abort
  --diag-warmup-chunks 10
  --diag-warmup-epochs 1
  --diag-min-val-samples 512
  --diag-require-consecutive 2'''
)

# 최대 재시도 횟수
MAX_ATTEMPTS = 3

# 허용 변경 키 및 검증 규칙
ALLOWED_KEYS = {
    # Class distribution / label policy
    "direction3-threshold": lambda v: isinstance(v, (int, float)) and 0.001 <= float(v) <= 0.02,
    # Temporal horizon and input length
    "horizon": lambda v: isinstance(v, int) and 5 <= v <= 60,
    "seq-len": lambda v: isinstance(v, int) and 20 <= v <= 200,
    # Optimizer settings
    "lr": lambda v: isinstance(v, (int, float)) and 1e-5 <= float(v) <= 3e-3,
    "weight-decay": lambda v: isinstance(v, (int, float)) and 0.0 <= float(v) <= 1e-2,
    "batch-size": lambda v: isinstance(v, int) and 32 <= v <= 512,
    # Classification loss shaping (direction3)
    "label-smoothing": lambda v: isinstance(v, (int, float)) and 0.0 <= float(v) <= 0.3,
    "focal-gamma": lambda v: isinstance(v, (int, float)) and 0.0 <= float(v) <= 5.0,
    "loss": lambda v: isinstance(v, str) and v in {"ce", "focal", "mse", "huber"},
}

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def _list_openai_models() -> set[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[오류] OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 모델 목록을 가져올 수 없습니다.")
        sys.exit(4)
    try:
        resp = requests.get("https://api.openai.com/v1/models", headers={
            "Authorization": f"Bearer {api_key}",
        }, timeout=30)
        if resp.status_code >= 400:
            print(f"[오류] 모델 목록 조회 실패: {resp.status_code} {resp.text[:800]}")
            sys.exit(4)
        data = resp.json()
        ids = {item.get("id", "") for item in data.get("data", [])}
        return ids
    except Exception as e:
        print(f"[오류] 모델 목록 조회 중 예외: {e}")
        sys.exit(4)


def get_openai_model() -> str:
    """환경변수 OPENAI_MODEL을 읽고, OpenAI 모델 목록과 대조하여 없으면 강제 종료."""
    mdl = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    available = _list_openai_models()
    if mdl not in available:
        # 상위 몇 개 예시만 출력
        sample = list(sorted(available))[:10]
        print(f"[오류] 지정된 모델 '{mdl}'을(를) OpenAI 모델 목록에서 찾을 수 없습니다.")
        if sample:
            print(f"[정보] 사용 가능한 모델 일부: {sample} ...")
        print("[조치] .env의 OPENAI_MODEL을 사용 가능한 모델로 변경한 뒤 다시 실행하세요.")
        sys.exit(4)
    return mdl


def run_training(cmd: List[str]) -> subprocess.CompletedProcess:
    print("\n[정보] 학습 실행:")
    print(" ", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("\n[학습 STDOUT]\n" + proc.stdout[-4000:])
    if proc.stderr:
        print("\n[학습 STDERR]\n" + proc.stderr[-2000:])
    print(f"\n[정보] 종료 코드: {proc.returncode}")
    return proc


def cmd_list_to_kv(cmd: List[str]) -> Dict[str, Any]:
    # 매우 단순한 파서: --key value 형태만 처리
    out: Dict[str, Any] = {}
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and (i + 1) < len(cmd) and not cmd[i + 1].startswith("--"):
            key = tok[2:]
            val = cmd[i + 1]
            # 숫자형 변환 시도
            try:
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except Exception:
                out[key] = val
            i += 2
        else:
            i += 1
    return out


def kv_to_cmd(base: List[str], kv: Dict[str, Any]) -> List[str]:
    # base를 복사 후, --key value 쌍을 업데이트
    out = base[:]
    idx = 0
    while idx < len(out):
        if out[idx].startswith("--") and (idx + 1) < len(out) and not out[idx + 1].startswith("--"):
            key = out[idx][2:]
            if key in kv:
                out[idx + 1] = str(kv[key])
        idx += 1
    # 만약 base에 없는 허용 키가 있다면 추가
    for k, v in kv.items():
        flag = f"--{k}"
        if flag not in out:
            out.extend([flag, str(v)])
    return out


# ---------------- 메모리 관리 ----------------
def load_memory() -> Dict[str, Any]:
    try:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"interactions": []}


def save_memory(mem: Dict[str, Any]) -> None:
    try:
        with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[경고] 메모리 저장 실패: {e}")


def summarize_history(mem: Dict[str, Any]) -> str:
    if not mem or not mem.get("interactions"):
        return "(이전 대화 없음)"
    lines: list[str] = []
    for item in mem["interactions"][-5:]:
        attempt = item.get("attempt")
        applied_kv = item.get("applied_kv", {})
        changed = ", ".join([f"{k}={v}" for k, v in applied_kv.items()]) if applied_kv else "(적용 없음)"
        lines.append(f"- 시도#{attempt}: 적용 {changed}")
    return "\n".join(lines)


def already_tried_pairs(mem: Dict[str, Any]) -> set[tuple]:
    tried: set[tuple] = set()
    for item in mem.get("interactions", []):
        for k, v in item.get("applied_kv", {}).items():
            tried.add((k, v))
    return tried


def build_prompt(stdout_tail: str, stderr_tail: str, current_kv: Dict[str, Any], history_summary: str, tried_pairs: List[str]) -> str:
    return textwrap.dedent(f"""
    역할: 당신은 머신러닝 트레이닝 자동화 도우미입니다. 아래 로그를 분석해 실패 원인을 간단히 요약하고,
    허용된 범위 내에서 CLI 하이퍼파라미터를 수정해 다음 재시도에 사용할 JSON을 제공합니다.

    제약:
    - 반드시 JSON만 출력하세요. 다른 텍스트는 금지.
    - JSON 스키마:
      {{
        "suggested_args": {{
          "direction3-threshold"?: float,
          "horizon"?: int,
          "seq-len"?: int,
          "lr"?: float,
          "weight-decay"?: float,
          "batch-size"?: int,
          "label-smoothing"?: float,
          "focal-gamma"?: float,
          "loss"?: "ce" | "focal" | "mse" | "huber"
        }},
        "rationale": "한글 설명",
        "changed_keys": ["..."]
      }}
    - 허용 키만 제안: {list(ALLOWED_KEYS.keys())}
    - 한 번에 최대 2개 키만 변경하세요.
    - 권장 범위: threshold[0.001,0.02], horizon[5,60], seq-len[20,200], lr[1e-5,3e-3], weight-decay[0,1e-2], batch-size[32,512], label-smoothing[0.0,0.3], focal-gamma[0.0,5.0], loss∈[ce,focal,mse,huber]
    - 아래의 '이미 시도한 변경'과 동일한 제안은 피하세요. 동일한 (key,value) 조합을 다시 제안하지 마세요.

    현재 CLI 값:
    {json.dumps(current_kv, ensure_ascii=False)}

    이미 시도한 변경 요약:
    {history_summary}

    이미 시도한 (key,value) 목록:
    {json.dumps(tried_pairs, ensure_ascii=False)}

    최근 STDOUT:
    ```
    {stdout_tail}
    ```

    최근 STDERR:
    ```
    {stderr_tail}
    ```
    """)


def ask_chatgpt(prompt: str) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[오류] OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 재학습 자동 조정을 사용할 수 없습니다.")
        return {}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    model_name = get_openai_model()
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful ML training assistant."},
            {"role": "user", "content": prompt},
        ]
    }
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code >= 400:
            print(f"[오류] OpenAI API 응답 코드 {resp.status_code}. 본문: {resp.text[:1000]}")
            sys.exit(5)
        # 정상 응답 처리
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # JSON만 오도록 요구했으나 방어적으로 처리
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end+1]
        return json.loads(content)
    except Exception as e:
        print(f"[오류] OpenAI API 호출 실패: {e}")
        sys.exit(5)


def apply_suggestions(current_kv: Dict[str, Any], suggestions: Dict[str, Any], mem: Dict[str, Any]) -> Dict[str, Any]:
    sug_args = suggestions.get("suggested_args", {}) if suggestions else {}
    print("\n[정보] ChatGPT 제안:")
    print(json.dumps(suggestions, ensure_ascii=False, indent=2))

    # 필터링: 허용 키와 유효성 검사
    new_kv = dict(current_kv)
    applied = []
    tried = already_tried_pairs(mem)
    for k, v in sug_args.items():
        if k in ALLOWED_KEYS and ALLOWED_KEYS[k](v):
            # 중복 (key,value) 시도 방지
            if (k, v) in tried:
                continue
            new_kv[k] = v if not isinstance(v, float) else float(v)
            applied.append(k)
    # 2개 초과 변경은 제한
    if len(applied) > 2:
        applied = applied[:2]
        new_kv = {**current_kv, **{k: new_kv[k] for k in applied}}

    if not applied:
        print("[정보] 적용 가능한 변경이 없습니다. 기본 휴리스틱을 적용합니다.")
        # 기본 휴리스틱: threshold ↓ 또는 horizon ↑
        thr = float(current_kv.get("direction3-threshold", 0.01))
        if thr > 0.003:
            new_kv["direction3-threshold"] = max(0.003, thr / 2)
            applied = ["direction3-threshold"]
        else:
            hz = int(current_kv.get("horizon", 20))
            new_kv["horizon"] = min(60, hz + 10)
            applied = ["horizon"]

    print(f"[정보] 적용된 변경 키: {applied}")
    return new_kv


def main():
    attempt = 1
    mem = load_memory()
    cmd = BASE_CMD
    # 메모리를 토대로 기본 CLI 초기화 (가장 최근 적용값 반영)
    def _get_last_applied_kv(m: Dict[str, Any]) -> Dict[str, Any]:
        for item in reversed(m.get("interactions", [])):
            akv = item.get("applied_kv", {})
            if akv:
                return akv
        return {}
    init_kv = _get_last_applied_kv(mem)
    if init_kv:
        # 허용 키만 필터링
        init_kv = {k: init_kv[k] for k in init_kv.keys() if k in ALLOWED_KEYS}
        if init_kv:
            print(f"[정보] 메모리 기반 초기 CLI 적용: {init_kv}")
            cmd = kv_to_cmd(cmd, init_kv)
    while attempt <= MAX_ATTEMPTS:
        print("\n" + "="*20 + f" 시도 #{attempt} " + "="*20)
        proc = run_training(cmd)
        if proc.returncode == 0:
            print("[성공] 학습이 정상 종료되었습니다.")
            sys.exit(0)
        if proc.returncode != 2:
            print("[정보] 자동 진단으로 인한 종료가 아닙니다. 자동 재시도를 중단합니다.")
            sys.exit(proc.returncode)

        # 진단 실패(코드 2): 최근 로그 요약 + ChatGPT 질의
        stdout_tail = proc.stdout[-2000:]
        stderr_tail = proc.stderr[-1000:]
        current_kv = cmd_list_to_kv(cmd)
        hist_summary = summarize_history(mem)
        tried_pairs = [f"{k}={v}" for (k, v) in sorted(list(already_tried_pairs(mem)))]
        prompt = build_prompt(stdout_tail, stderr_tail, current_kv, hist_summary, tried_pairs)
        suggestions = ask_chatgpt(prompt)
        new_kv = apply_suggestions(current_kv, suggestions, mem)
        cmd = kv_to_cmd(cmd, new_kv)

        # 메모리에 기록
        applied_kv = {k: new_kv.get(k) for k in ALLOWED_KEYS.keys() if k in new_kv and current_kv.get(k) != new_kv.get(k)}
        mem.setdefault("interactions", []).append({
            "attempt": attempt,
            "prompt": prompt,
            "response": suggestions,
            "applied_kv": applied_kv,
        })
        save_memory(mem)

        print("\n[정보] 수정된 CLI로 재시작합니다:")
        print(" ", " ".join(cmd))
        attempt += 1
        time.sleep(3)

    print("[종료] 최대 재시도 횟수에 도달했습니다. 수동 점검이 필요합니다.")
    sys.exit(3)


if __name__ == "__main__":
    main()
