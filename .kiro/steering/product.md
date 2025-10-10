# Product Overview

Stock Bot is an AI-powered trading system for Korean stock market scalping using reinforcement learning. The system combines embedding models with GRPO (Group Relative Policy Optimization) for ultra-short-term trading decisions.

## Core Features

- **Embedding Models**: Transform market data sequences into meaningful representations
- **GRPO Agent**: Reinforcement learning agent for trading decisions (buy/hold/sell)
- **Scalping Focus**: Optimized for trades lasting under 30 seconds
- **Real-time Processing**: Sub-10ms inference latency for live trading
- **Backtesting**: Comprehensive evaluation with realistic transaction costs

## Performance Targets

- Win rate > 50%
- Sharpe ratio > 1.0
- Maximum drawdown < 10%
- Average holding time < 30 seconds
- Inference latency < 10ms

## Data Pipeline

The system processes Korean stock market order book data (호가 데이터) with features including:
- Price levels and volumes
- Market indicators (등락률, 체결강도, 거래회전율)
- Time-based features (normalized timestamps)
- Stock-specific embeddings

## Trading Costs

- Transaction costs: 0.215% (one-way)
- Slippage: 0.01% (one-way)
- Total round-trip cost: ~0.45%