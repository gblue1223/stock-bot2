import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv, TradingEnvConfig
from .policies import TimeSeriesCNNExtractor


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
    # New: reward/obs customization
    reward_mode: str = "delta",
    transaction_cost_bps: float = 0.0,
    holding_cost_bps: float = 0.0,
    obs_format: str = "flat",
    # Policy selection
    policy: str = "mlp",
    device: str = "cpu",
    # Analysis encoder
    encoder_ckpt: Optional[str] = None,
    # Scalping heuristics
    w_fast: float = 0.0,
    w_sticky: float = 0.0,
    priority_bonus: float = 0.0,
    activity_window: int = 5,
    stoploss_wait: int = 3,
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
    - reward_mode (str): 보상 방식 {delta, return, log_return}.
    - transaction_cost_bps (float): 거래 비용 (bp).
    - holding_cost_bps (float): 보유 비용 (bp, per-step).
    - obs_format (str): 관측 포맷 {flat, matrix}. matrix는 CNN extractor와 함께 권장.
    - policy (str): "mlp" 또는 "cnn". cnn은 TimeSeriesCNNExtractor 사용.
    - device (str): 학습 장치 (예: "cpu", "cuda").
    - encoder_ckpt (Optional[str]): 사전학습 분석모델 체크포인트 경로(디렉터리 또는 model.pt). 제공 시 상태 벡터를 관측으로 사용.
    - w_fast, w_sticky (float): 매수 우선순위 휴리스틱 가중치(활동성/점착성).
    - priority_bonus (float): 롱 포지션 오픈 시 우선순위 점수에 곱해지는 보너스 보상.
    - activity_window (int): 휴리스틱 계산에 사용하는 최근 관찰 윈도우(초 단위 스텝).
    - stoploss_wait (int): 진입 후 n초 내 상승 없으면 강제 손절.
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
        reward_mode=reward_mode,
        transaction_cost_bps=transaction_cost_bps,
        holding_cost_bps=holding_cost_bps,
        obs_format=("matrix" if policy == "cnn" else obs_format),
        encoder_ckpt=encoder_ckpt,
        w_fast=w_fast,
        w_sticky=w_sticky,
        priority_bonus=priority_bonus,
        activity_window=activity_window,
        stoploss_wait=stoploss_wait,
    )
    vec_env = DummyVecEnv([make_env(env_cfg)])

    AlgoClass = ALGOS[algo]

    policy_kwargs = {}
    policy_id = "MlpPolicy"
    if policy == "cnn" and encoder_ckpt is None:
        # Use custom extractor only when we are using raw windows (no encoder)
        # With DummyVecEnv and obs_format=matrix, obs space is (T, F)
        # SB3 default ActorCriticPolicy (MlpPolicy) can still be used with custom extractor
        policy_id = "MlpPolicy"
        # Need seq_len and feature count for flat case; for matrix obs, extractor can infer from space
        T, F = (seq_len, None)
        if env_cfg.obs_format == "matrix":
            # observation_space shape: (T, F)
            T, F = vec_env.observation_space.shape
        else:
            # Flat: need to specify both explicitly
            F = int(vec_env.observation_space.shape[0] // seq_len)
        policy_kwargs = {
            "features_extractor_class": TimeSeriesCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "seq_len": T,
                "num_features": F,
            },
        }

    if algo == "ppo":
        model = AlgoClass(
            policy_id,
            vec_env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=n_steps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )
    else:  # a2c
        model = AlgoClass(
            policy_id,
            vec_env,
            learning_rate=lr,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )

    model.learn(total_timesteps=total_timesteps)
    save_path = os.path.join(out_dir, f"{algo}_model")
    model.save(save_path)
    print(f"Saved RL model to {save_path}.zip")


def main():
    p = argparse.ArgumentParser(description="TradingEnv에서 PPO/A2C 학습")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo", help="사용할 RL 알고리즘 ('ppo' 또는 'a2c'). 기본값: ppo")
    p.add_argument("--db", default="datasets/datasets.db", help="데이터셋이 담긴 SQLite DB 파일 경로. 기본값: datasets/datasets.db")
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
    # New
    p.add_argument("--reward-mode", choices=["delta", "return", "log_return"], default="delta", help="보상 방식 선택")
    p.add_argument("--tc-bps", type=float, default=0.0, help="거래비용(bps). 예: 5 = 0.05%")
    p.add_argument("--hc-bps", type=float, default=0.0, help="보유비용(bps per step)")
    p.add_argument("--obs-format", choices=["flat", "matrix"], default="flat", help="관측 포맷: flat 또는 matrix")
    p.add_argument("--policy", choices=["mlp", "cnn"], default="mlp", help="정책 네트워크 유형")
    p.add_argument("--device", default="cpu", help="학습 장치: cpu/cuda")
    p.add_argument("--encoder-ckpt", default=None, help="사전학습 분석모델 체크포인트 경로(디렉터리 또는 model.pt). 제공 시 상태벡터 관측 사용")
    # Scalping heuristics
    p.add_argument("--w-fast", type=float, default=0.0, help="빠른 체결/활동성 휴리스틱 가중치")
    p.add_argument("--w-sticky", type=float, default=0.0, help="가격 점착성(하락 드뭄) 휴리스틱 가중치")
    p.add_argument("--priority-bonus", type=float, default=0.0, help="롱 포지션 오픈 시 우선순위 보상 스케일")
    p.add_argument("--activity-window", type=int, default=5, help="휴리스틱 계산을 위한 최근 윈도우 길이(스텝)")
    p.add_argument("--stoploss-wait", type=int, default=3, help="진입 후 n스텝 안 오르면 손절")
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
        reward_mode=args.reward_mode,
        transaction_cost_bps=args.tc_bps,
        holding_cost_bps=args.hc_bps,
        obs_format=args.obs_format,
        policy=args.policy,
        device=args.device,
        encoder_ckpt=args.encoder_ckpt,
        w_fast=args.w_fast,
        w_sticky=args.w_sticky,
        priority_bonus=args.priority_bonus,
        activity_window=args.activity_window,
        stoploss_wait=args.stoploss_wait,
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
