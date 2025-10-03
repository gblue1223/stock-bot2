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
- 현재 학습 코드는 손실 종류에 ce/mse/huber만 지원합니다. (focal 미지원)
- 안전한 변경만 적용합니다: direction3-threshold, horizon, lr, weight-decay, batch-size
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
# 기본 학습 CLI (사용자 예시 기반)
BASE_CMD: List[str] = shlex.split(
    r'''./.venv64/Scripts/python -m ai_trader.ml.train_supervised 
    --db "C:\\Users\\user\\Workspace\\datasets@20251002\\datasets_norm_all.duckdb" 
    --table datasets 
    --out models/supervised 
    --seq-len 60 
    --horizon 20 
    --chunk-size 50000 
    --device cuda 
    --batch-size 128 
    --epochs 10 
    --lr 3e-4 
    --weight-decay 0 
    --progress-every 10 
    --ckpt-every-chunks 50 
    --resume-from models/supervised/checkpoints 
    --target-col 현재가 
    --aux-task direction3 
    --loss ce 
    --direction3-threshold 0.015'''
)

# 최대 재시도 횟수
MAX_ATTEMPTS = 3

# 허용 변경 키 및 검증 규칙
ALLOWED_KEYS = {
    "direction3-threshold": lambda v: isinstance(v, (int, float)) and 0.001 <= float(v) <= 0.02,
    "horizon": lambda v: isinstance(v, int) and 5 <= v <= 60,
    "lr": lambda v: isinstance(v, (int, float)) and 1e-5 <= float(v) <= 3e-3,
    "weight-decay": lambda v: isinstance(v, (int, float)) and 0.0 <= float(v) <= 1e-2,
    "batch-size": lambda v: isinstance(v, int) and 32 <= v <= 512,
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


def build_prompt(stdout_tail: str, stderr_tail: str, current_kv: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""
    역할: 당신은 머신러닝 트레이닝 자동화 도우미입니다. 아래 로그를 분석해 실패 원인을 간단히 요약하고,
    허용된 범위 내에서 CLI 하이퍼파라미터를 수정해 다음 재시도에 사용할 JSON을 제공합니다.

    제약:
    - 반드시 JSON만 출력하세요. 다른 텍스트는 금지.
    - JSON 스키마:
      {{
        "suggested_args": {{ "direction3-threshold": float, "horizon": int, "lr": float, "weight-decay": float, "batch-size": int }},
        "rationale": "한글 설명",
        "changed_keys": ["..."]
      }}
    - 허용 키만 제안: {list(ALLOWED_KEYS.keys())}
    - 한 번에 최대 2개 키만 변경하세요.
    - 권장 범위: threshold[0.001,0.02], horizon[5,60], lr[1e-5,3e-3], weight-decay[0,1e-2], batch-size[32,512]

    현재 CLI 값:
    {json.dumps(current_kv, ensure_ascii=False)}

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


def apply_suggestions(current_kv: Dict[str, Any], suggestions: Dict[str, Any]) -> Dict[str, Any]:
    sug_args = suggestions.get("suggested_args", {}) if suggestions else {}
    changed_keys = suggestions.get("changed_keys", []) if suggestions else []
    print("\n[정보] ChatGPT 제안:")
    print(json.dumps(suggestions, ensure_ascii=False, indent=2))

    # 필터링: 허용 키와 유효성 검사
    new_kv = dict(current_kv)
    applied = []
    for k, v in sug_args.items():
        if k in ALLOWED_KEYS and ALLOWED_KEYS[k](v):
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
    cmd = BASE_CMD
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
        prompt = build_prompt(stdout_tail, stderr_tail, current_kv)
        suggestions = ask_chatgpt(prompt)
        new_kv = apply_suggestions(current_kv, suggestions)
        cmd = kv_to_cmd(cmd, new_kv)

        print("\n[정보] 수정된 CLI로 재시작합니다:")
        print(" ", " ".join(cmd))
        attempt += 1
        time.sleep(3)

    print("[종료] 최대 재시도 횟수에 도달했습니다. 수동 점검이 필요합니다.")
    sys.exit(3)


if __name__ == "__main__":
    main()
