# Product Overview

Stock Bot2 is an AI-powered trading system that combines machine learning and reinforcement learning for Korean stock market trading through the Kiwoom API.

## Core Components

- **AI Trader**: ML/RL pipeline for supervised learning (CNN+LSTM+Attention) and reinforcement learning (PPO/A2C) using DuckDB datasets
- **Koapys**: REST API client wrapper for Kiwoom trading API with compatibility layer for existing koapys interfaces  
- **Data Pipeline**: Scripts for dataset generation, normalization, and merging from trading data
- **Utilities**: Common libraries for event handling, I/O operations, plotting, and Windows integration

## Key Features

- Supervised learning for price prediction using time series data
- Reinforcement learning trading environments with customizable reward functions
- Scalping strategies with stop-loss mechanisms and activity heuristics
- Real-time trading integration through Kiwoom REST API
- Comprehensive data processing and normalization tools