---
inclusion: always
---

# Product Overview

Stock-bot is an AI-powered Korean stock trading system that combines supervised learning and reinforcement learning for automated trading decisions. The system processes real-time market data (order book, execution, trader information) from Kiwoom Securities API to train ML models and execute trading strategies.

## Core Components

- **Data Pipeline**: Ingests and normalizes CSV market data into DuckDB databases with monthly sharding
- **ML Training**: CNN+LSTM+Attention models for price prediction and direction classification
- **RL Trading**: PPO/A2C agents with custom trading environments supporting scalping strategies
- **API Integration**: REST-based Kiwoom Securities client for live trading operations

## Key Features

- Multi-feature time series analysis (60+ market indicators)
- 3-class direction prediction (up/neutral/down) with configurable thresholds
- Scalping-optimized RL environment with stop-loss and activity heuristics
- Checkpoint-based resumable training with TensorBoard monitoring
- Monthly-sharded DuckDB storage for efficient data management
