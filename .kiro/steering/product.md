# Product Overview

AI-powered Korean stock market scalping system combining AutoEncoder-based embeddings with GRPO (Group Relative Policy Optimization) reinforcement learning for ultra-fast trading decisions.

## Core Capabilities

- **Ultra-fast inference**: 2.87ms real-time trading decisions (3x faster than 10ms target)
- **Scalping-focused**: Optimized for sub-30-second trades in Korean stock market
- **Self-supervised learning**: AutoEncoder-based embeddings with 10-100x faster training than contrastive methods
- **GRPO reinforcement learning**: Group relative policy optimization for stable learning across market conditions
- **Large-scale data**: Capable of training on 1.5 billion data points within weeks

## System Architecture

1. **AutoEncoder Pre-training**: Self-supervised learning on time-series market data
2. **Fine-tuning**: Task-specific adaptation for trading signals (buy/hold/sell)
3. **GRPO Training**: Reinforcement learning with group-relative advantages
4. **Real-time Inference**: Sub-3ms prediction pipeline for live trading

## Target Market

Korean stock market (KRX) with focus on high-frequency scalping strategies during market hours (09:00-11:00).
