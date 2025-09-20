import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_trader.rl.env import TradingEnv, TradingEnvConfig

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
}


@dataclass
class PipelineConfig:
    algo: str = "ppo"
    db: str = "datasets/datasets.db"
    table: str = "datasets"
    code: Optional[str] = None
    date: Optional[str] = None
    seq_len: int = 60
    target_col: Optional[str] = None
    scale_obs: bool = False
    encoder_ckpt: Optional[str] = None
    rl_model_path: str = "models/rl/ppo_model"


def run_pipeline(cfg: PipelineConfig):
    algo = cfg.algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}. Choose from {list(ALGOS.keys())}")

    env_cfg = TradingEnvConfig(
        db_path=cfg.db,
        table=cfg.table,
        code=cfg.code,
        date=cfg.date,
        seq_len=cfg.seq_len,
        target_col=cfg.target_col,
        scale_obs=cfg.scale_obs,
        encoder_ckpt=cfg.encoder_ckpt,
    )
    env = DummyVecEnv([lambda: TradingEnv(env_cfg)])

    AlgoClass = ALGOS[algo]
    model = AlgoClass.load(cfg.rl_model_path, env=env, device="cpu")

    obs = env.reset()
    cum_reward = 0.0
    step = 0
    print("Starting pipeline run...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        r = float(rewards[0] if hasattr(rewards, "__len__") else rewards)
        d = bool(dones[0] if hasattr(dones, "__len__") else dones)
        cum_reward += r
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        print(f"step={step:06d} action={int(action[0]) if hasattr(action,'__len__') else int(action)} reward={r:.6f} pnl={info.get('pnl', 0.0):.6f} pos={info.get('position', 0)}")
        step += 1
        if d:
            break
    print(f"Finished. Cumulative reward (approx PnL): {cum_reward:.6f}")


def main():
    p = argparse.ArgumentParser(description="Integration pipeline: data -> encoder -> PPO action")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--db", default="datasets/datasets.db")
    p.add_argument("--table", default="datasets")
    p.add_argument("--code", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--target-col", default=None)
    p.add_argument("--scale-obs", action="store_true")
    p.add_argument("--encoder-ckpt", default=None, help="사전학습 분석모델 체크포인트 경로(디렉터리 또는 model.pt)")
    p.add_argument("--model", default="models/rl/ppo_model", help="RL 모델 경로(압축 확장자 없는 경로)")
    args = p.parse_args()

    run_pipeline(PipelineConfig(
        algo=args.algo,
        db=args.db,
        table=args.table,
        code=args.code,
        date=args.date,
        seq_len=args.seq_len,
        target_col=args.target_col,
        scale_obs=args.scale_obs,
        encoder_ckpt=args.encoder_ckpt,
        rl_model_path=args.model,
    ))


if __name__ == "__main__":
    main()
