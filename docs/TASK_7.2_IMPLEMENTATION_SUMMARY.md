# Task 7.2 Implementation Summary: 롤아웃 수집 구현

## Overview
Task 7.2 implements rollout collection for the GRPO algorithm, enabling the trainer to execute multiple episodes with the current policy and collect trajectory data for policy optimization.

## Implementation Details

### Core Method: `collect_rollouts()`
Location: `ai_trader/grpo/grpo.py`

The `collect_rollouts()` method implements the following functionality:

1. **Episode Execution** (요구사항 4.1)
   - Executes multiple episodes (configurable: 8-16 per group)
   - Uses current policy to sample actions in stochastic mode
   - Runs episodes until termination or truncation

2. **Data Collection**
   - **States**: Embedding vectors from the environment
   - **Actions**: Sampled from policy (0=hold, 1=buy, 2=sell)
   - **Rewards**: Calculated by environment based on trading performance
   - **Next States**: Subsequent embedding vectors
   - **Dones**: Episode termination flags
   - **Log Probabilities**: Action log probabilities for policy gradient

3. **Episode Metadata Collection**
   - Episode reward (total return)
   - Episode steps (number of timesteps)
   - Trading metrics (from environment):
     - Number of trades
     - Average holding time
     - Sharpe ratio
     - Quick exit rule violations
     - Win rate
     - Average profit per trade

### Action Sampling: `_sample_action()`
The method intelligently handles action sampling:

- **With Policy**: If policy has `get_action()` method, uses it for stochastic sampling
- **Without Policy**: Falls back to random actions with uniform distribution (for testing/initialization)
- Returns both action and log probability for policy gradient computation

### Data Structure
Each episode is stored as a dictionary:
```python
{
    'states': np.ndarray,           # (T, embedding_dim)
    'actions': np.ndarray,          # (T,)
    'rewards': np.ndarray,          # (T,)
    'next_states': np.ndarray,      # (T, embedding_dim)
    'dones': np.ndarray,            # (T,)
    'log_probs': np.ndarray,        # (T,)
    'metadata': dict                # Episode-level metrics
}
```

## Requirements Verification

### ✅ Requirement 4.1: Rollout Collection
- [x] Execute multiple episodes with current policy (8-16 per group)
- [x] Store states, actions, rewards, next states
- [x] Collect episode metadata
- [x] Track total timesteps across all rollouts

### Key Features
1. **Flexible Episode Count**: Configurable number of episodes per rollout
2. **Timestep Tracking**: Maintains cumulative timestep counter
3. **Comprehensive Logging**: Debug logs for episode progress
4. **Metadata Integration**: Captures all environment-provided metrics

## Testing

### Test Coverage
Location: `tests/test_grpo_rollout.py`

1. **test_collect_rollouts_basic**: Verifies basic rollout collection structure
2. **test_collect_rollouts_metadata**: Validates metadata collection
3. **test_collect_rollouts_timesteps**: Checks timestep counting
4. **test_sample_action_with_policy**: Tests action sampling with policy
5. **test_sample_action_without_policy**: Tests fallback to random actions
6. **test_collect_rollouts_multiple_batches**: Verifies multiple rollout batches

### Test Results
```
tests/test_grpo_rollout.py::test_collect_rollouts_basic PASSED
tests/test_grpo_rollout.py::test_collect_rollouts_metadata PASSED
tests/test_grpo_rollout.py::test_collect_rollouts_timesteps PASSED
tests/test_grpo_rollout.py::test_sample_action_with_policy PASSED
tests/test_grpo_rollout.py::test_sample_action_without_policy PASSED
tests/test_grpo_rollout.py::test_collect_rollouts_multiple_batches PASSED

6 passed in 2.03s
```

## Integration with GRPO Pipeline

The `collect_rollouts()` method integrates seamlessly with the GRPO training loop:

```python
# Training iteration
episodes = trainer.collect_rollouts(num_episodes=48)  # 4 groups × 12 episodes
grouped_episodes = trainer.group_episodes(episodes)   # Task 7.3
advantages = trainer.compute_group_relative_advantages(grouped_episodes)  # Task 7.4
trainer.update_policy(episodes, advantages)  # Task 7.5
```

## Future Enhancements

1. **Parallel Rollout Collection**: Use vectorized environments for faster collection
2. **Prioritized Sampling**: Sample episodes based on importance
3. **Replay Buffer**: Store and reuse rollouts across multiple updates
4. **On-Policy vs Off-Policy**: Support both modes with appropriate handling

## Dependencies

- **Environment**: `GRPOScalpingEnv` (Task 6.x - ✅ Complete)
- **Policy**: `GRPOPolicy` (Task 8.1 - ⏳ Pending)
  - Current implementation uses fallback for testing
  - Will integrate seamlessly once policy is implemented

## Conclusion

Task 7.2 is **COMPLETE** and ready for integration with subsequent tasks (7.3, 7.4, 7.5). The implementation:
- ✅ Meets all requirements from specification
- ✅ Passes all unit tests
- ✅ Provides comprehensive data collection
- ✅ Integrates with existing environment
- ✅ Supports future policy implementation
- ✅ Includes proper logging and error handling

The rollout collection forms the foundation for GRPO's group-based policy optimization, enabling the algorithm to learn effective scalping strategies through comparative performance analysis.
