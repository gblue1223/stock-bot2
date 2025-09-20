import argparse
import os

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv, TradingEnvConfig

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
}


def run(
    algo: str,
    db: str,
    model_path: str,
    table: str = "datasets",
    code: str | None = None,
    date: str | None = None,
    seq_len: int = 60,
    target_col: str | None = None,
    scale_obs: bool = False,
    encoder_ckpt: str | None = None,
    # Heuristic options (round-figure removed)
    w_fast: float = 0.0,
    w_sticky: float = 0.0,
    priority_bonus: float = 0.0,
    activity_window: int = 5,
    stoploss_wait: int = 3,
):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}. Choose from {list(ALGOS.keys())}")

    env_cfg = TradingEnvConfig(
        db_path=db,
        table=table,
        code=code,
        date=date,
        seq_len=seq_len,
        target_col=target_col,
        scale_obs=scale_obs,
        encoder_ckpt=encoder_ckpt,
        w_fast=w_fast,
        w_sticky=w_sticky,
        priority_bonus=priority_bonus,
        activity_window=activity_window,
        stoploss_wait=stoploss_wait,
    )
    env = DummyVecEnv([lambda: TradingEnv(env_cfg)])

    AlgoClass = ALGOS[algo]
    model = AlgoClass.load(model_path, env=env, device="cpu")

    # Note: DummyVecEnv.reset() returns only obs
    obs = env.reset()
    cum_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        # DummyVecEnv.step returns (obs, rewards, dones, infos)
        obs, rewards, dones, infos = env.step(action)
        # rewards/dones are vectorized (shape [n_envs])
        r = float(rewards[0] if hasattr(rewards, "__len__") else rewards)
        d = bool(dones[0] if hasattr(dones, "__len__") else dones)
        cum_reward += r
        if d:
            break
    print(f"Episode cumulative reward (approx PnL): {cum_reward:.6f}")


def main():
    p = argparse.ArgumentParser(description="Run RL inference on TradingEnv")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--db", default="datasets/datasets.db")
    p.add_argument("--model", default="models/rl/ppo_model")
    p.add_argument("--table", default="datasets")
    p.add_argument("--code", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--target-col", default=None)
    p.add_argument("--scale-obs", action="store_true")
    p.add_argument("--encoder-ckpt", default=None, help="사전학습 분석모델 체크포인트 경로(디렉터리 또는 model.pt)")
    # Heuristic flags (round-figure removed)
    p.add_argument("--w-fast", type=float, default=0.0)
    p.add_argument("--w-sticky", type=float, default=0.0)
    p.add_argument("--priority-bonus", type=float, default=0.0)
    p.add_argument("--activity-window", type=int, default=5)
    p.add_argument("--stoploss-wait", type=int, default=3)
    args = p.parse_args()

    run(
        algo=args.algo,
        db=args.db,
        model_path=args.model,
        table=args.table,
        code=args.code,
        date=args.date,
        seq_len=args.seq_len,
        target_col=args.target_col,
        scale_obs=args.scale_obs,
        encoder_ckpt=args.encoder_ckpt,
        w_fast=args.w_fast,
        w_sticky=args.w_sticky,
        priority_bonus=args.priority_bonus,
        activity_window=args.activity_window,
        stoploss_wait=args.stoploss_wait,
    )


if __name__ == "__main__":
    main()
