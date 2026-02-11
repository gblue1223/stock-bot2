import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from ai_trader.grpo.environments.scalping_env_v2 import GRPOScalpingEnvV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_v2_env():
    parquet_path = r"C:\Users\user\Workspace\datasets@20260117\embeddings_v2"
    db_path = r"C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb"
    
    logger.info("Initializing V2 Environment...")
    env = GRPOScalpingEnvV2(
        parquet_path=parquet_path,
        db_path=db_path,
        table_name='datasets',
        embedding_dim=128,
        transaction_cost_rate=0.00215,
        max_episode_steps=100
    )
    
    logger.info("Resetting Environment...")
    obs, info = env.reset()
    logger.info(f"Initial Observation Shape: {obs.shape}")
    logger.info(f"Info: {info}")
    
    # Check observation content (should not be all zeros)
    if np.all(obs == 0):
        logger.warning("Observation is all zeros!")
    else:
        logger.info("Observation contains data.")
        
    # Run a few steps
    logger.info("Running simulation steps...")
    total_reward = 0
    for i in range(10):
        action = 1 if i == 0 else 0 # Buy on first step
        if i == 5: action = 2 # Sell on 5th step
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        logger.info(f"Step {i}: Action={action}, Reward={reward:.4f}, Price={info['price']:.2f}, Time={info['time']}")
        
        if terminated or truncated:
            break
            
    logger.info(f"Total Reward: {total_reward}")
    logger.info("Test Complete.")

if __name__ == "__main__":
    test_v2_env()
