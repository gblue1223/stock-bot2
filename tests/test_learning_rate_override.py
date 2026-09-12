import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.grpo import GRPOTrainer


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


def test_learning_rate_override_preserves_adam_and_progress(tmp_path):
    policy = torch.nn.Linear(2, 1)
    trainer = GRPOTrainer(policy, object())
    trainer.optimizer = torch.optim.Adam([
        {'params': [policy.weight], 'lr': 3e-5, 'weight_decay': .01},
        {'params': [policy.bias], 'lr': 2e-5, 'betas': (.8, .95)},
    ])
    for _ in range(2):
        trainer.optimizer.zero_grad()
        policy(torch.ones(4, 2)).square().mean().backward()
        trainer.optimizer.step()
    trainer.total_timesteps, trainer.num_updates = 200, 19
    trainer.training_control_state.validation_count = 3
    trainer.extra_checkpoint_state = {'training_config': {'lr': 3e-5, 'gamma': 1.0}}
    before = copy.deepcopy(trainer.optimizer.state_dict())
    parameters = copy.deepcopy(policy.state_dict())
    control = copy.deepcopy(trainer.training_control_state)
    assert trainer.set_learning_rate(1e-5) == 1e-5
    expected = copy.deepcopy(before)
    for group in expected['param_groups']:
        group['lr'] = 1e-5
    assert_state_equal(trainer.optimizer.state_dict(), expected)
    assert_state_equal(policy.state_dict(), parameters)
    assert trainer.training_control_state == control
    assert (trainer.total_timesteps, trainer.num_updates) == (200, 19)
    assert trainer.learning_rate == 1e-5
    assert trainer.extra_checkpoint_state['training_config'] == {'lr': 1e-5, 'gamma': 1.0}
    destination = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(str(destination), 19, extra_state=trainer.extra_checkpoint_state)
    saved = torch.load(destination, weights_only=True)
    assert saved['config']['learning_rate'] == 1e-5
    assert saved['extra_state']['training_config']['lr'] == 1e-5
    assert_state_equal(saved['optimizer_state_dict'], expected)


@pytest.mark.parametrize('value', [0, -1e-5, float('nan'), float('inf'), -float('inf'),
                                  True, np.bool_(True), '1e-5', None])
def test_invalid_learning_rate_cannot_mutate_optimizer(value):
    trainer = GRPOTrainer(torch.nn.Linear(1, 1), object())
    before = copy.deepcopy(trainer.optimizer.state_dict())
    with pytest.raises(ValueError, match='finite positive'):
        trainer.set_learning_rate(value)
    assert_state_equal(trainer.optimizer.state_dict(), before)
    assert trainer.learning_rate == 3e-4
