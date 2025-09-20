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
    )


if __name__ == "__main__":
    main()
