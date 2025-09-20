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
    p = argparse.ArgumentParser(description="Train PPO/A2C on TradingEnv")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--db", default="datasets/datasets.db")
    p.add_argument("--out", default="models/rl")
    p.add_argument("--table", default="datasets")
    p.add_argument("--code", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--target-col", default=None)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--scale-obs", action="store_true")
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


if __name__ == "__main__":
    main()
