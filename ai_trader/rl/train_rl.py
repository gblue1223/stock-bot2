import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv, TradingEnvConfig


ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
}


def make_env(cfg: TradingEnvConfig):
    return lambda: TradingEnv(cfg)


def train(
    algo: str,
    db: str,
    out_dir: str,
    table: str = "datasets",
    code: Optional[str] = None,
    date: Optional[str] = None,
    seq_len: int = 60,
    target_col: Optional[str] = None,
    total_timesteps: int = 200_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    n_steps: int = 2048,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_steps: Optional[int] = None,
    scale_obs: bool = False,
):
    """
    TradingEnv에서 PPO 또는 A2C 알고리즘을 학습합니다.

    Parameters
    - algo (str): 사용할 알고리즘. {"ppo", "a2c"} 중 하나.
    - db (str): 데이터셋이 저장된 SQLite DB 파일 경로.
    - out_dir (str): 학습이 완료된 모델을 저장할 출력 디렉터리.
      저장 파일명은 "{algo}_model.zip" 형식으로 저장됩니다.
    - table (str, default="datasets"): DB에서 읽어올 테이블명.
    - code (Optional[str], default=None): 특정 종목 코드(예: "005930")로
      데이터셋을 필터링. None이면 환경 로직에 따라 전체 사용.
    - date (Optional[str], default=None): 특정 일자(YYYYMMDD)로 데이터셋을 필터링.
      None이면 환경 로직에 따라 전체 사용.
    - seq_len (int, default=60): 관측에 사용할 시퀀스(윈도우) 길이.
    - target_col (Optional[str], default=None): 환경/보상 계산에 참조될 수 있는
      타깃 컬럼명(환경 구현에 따라 사용).
    - total_timesteps (int, default=200_000): model.learn()에 전달되는 전체 학습 스텝 수.
    - lr (float, default=3e-4): 학습률(learning rate).
    - gamma (float, default=0.99): 감가율(discount factor).
    - n_steps (int, default=2048): PPO에서 사용하는 rollout 길이. A2C에서는 무시됩니다.
    - ent_coef (float, default=0.0): 탐색성 향상을 위한 엔트로피 보상 계수.
    - vf_coef (float, default=0.5): 가치함수 손실 가중치.
    - max_steps (Optional[int], default=None): 에피소드당 최대 스텝 수 또는
      데이터 순회 제한(환경 구현에 따라 의미가 달라질 수 있음).
    - scale_obs (bool, default=False): 관측치 스케일링/정규화 사용 여부.

    Returns
    - None. 학습된 모델을 디스크에 저장합니다.
    """
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}. Choose from {list(ALGOS.keys())}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    env_cfg = TradingEnvConfig(
        db_path=db,
        table=table,
        code=code,
        date=date,
        seq_len=seq_len,
        target_col=target_col,
        max_steps=max_steps,
        scale_obs=scale_obs,
    )
    vec_env = DummyVecEnv([make_env(env_cfg)])

    AlgoClass = ALGOS[algo]
    if algo == "ppo":
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=n_steps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=1,
        )
    else:  # a2c
        model = AlgoClass(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps)
    save_path = os.path.join(out_dir, f"{algo}_model")
    model.save(save_path)
    print(f"Saved RL model to {save_path}.zip")


def main():
    p = argparse.ArgumentParser(description="TradingEnv에서 PPO/A2C 학습")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo", help="사용할 RL 알고리즘 ('ppo' 또는 'a2c'). 기본값: ppo")
    p.add_argument("--db", default="datasets/datasets.db", help="데이터셋이 담긴 SQLite DB 경로. 기본값: datasets/datasets.db")
    p.add_argument("--out", default="models/rl", help="학습된 모델을 저장할 디렉터리({algo}_model.zip). 기본값: models/rl")
    p.add_argument("--table", default="datasets", help="DB에서 사용할 테이블명. 기본값: datasets")
    p.add_argument("--code", default=None, help="특정 종목 코드로 필터링 (예: 005930). 미지정 시 환경 로직에 따름")
    p.add_argument("--date", default=None, help="특정 일자(YYYYMMDD)로 필터링. 미지정 시 환경 로직에 따름")
    p.add_argument("--seq-len", type=int, default=60, help="관측 시퀀스(윈도우) 길이. 기본값: 60")
    p.add_argument("--target-col", default=None, help="환경/보상 로직에서 참조할 수 있는 타깃 컬럼명(환경 의존)")
    p.add_argument("--total-timesteps", type=int, default=200_000, help="학습에 사용할 총 타임스텝 수(model.learn). 기본값: 200000")
    p.add_argument("--lr", type=float, default=3e-4, help="학습률(learning rate). 기본값: 3e-4")
    p.add_argument("--gamma", type=float, default=0.99, help="감가율(discount factor) gamma. 기본값: 0.99")
    p.add_argument("--n-steps", type=int, default=2048, help="PPO의 rollout 길이. A2C에서는 무시. 기본값: 2048")
    p.add_argument("--ent-coef", type=float, default=0.0, help="엔트로피 보상 계수(탐색성). 기본값: 0.0")
    p.add_argument("--vf-coef", type=float, default=0.5, help="가치함수 손실 가중치. 기본값: 0.5")
    p.add_argument("--max-steps", type=int, default=None, help="에피소드 최대 스텝 수 또는 순회 제한(환경 의존). 기본값: None")
    p.add_argument("--scale-obs", action="store_true", help="관측치 스케일링/정규화 사용")
    args = p.parse_args()

    train(
        algo=args.algo,
        db=args.db,
        out_dir=args.out,
        table=args.table,
        code=args.code,
        date=args.date,
        seq_len=args.seq_len,
        target_col=args.target_col,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        gamma=args.gamma,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_steps=args.max_steps,
        scale_obs=args.scale_obs,
    )


#
# python -m ai_trader.ml.train_supervised --db models/test_datasets.db --table datasets --code 005930 --date 20240501 --out models/supervised
# python -m ai_trader.ml.train_supervised --db models/test_datasets2.db --table datasets --out models/supervised --seq-len 60 --horizon 1 --target-col 현재가 # 특정 종목/날짜로 제한
# python -m ai_trader.ml.train_supervised --db models/datasets.db --table datasets --out models/supervised --seq-len 60 --horizon 1 --target-col 현재가
#
# python -m ai_trader.ml.infer_supervised --db models/test_datasets2.db --ckpt models/supervised --top-k 10
# python -m ai_trader.ml.infer_supervised --db models/datasets.db --ckpt models/supervised --top-k 10
#
if __name__ == "__main__":
    main()
