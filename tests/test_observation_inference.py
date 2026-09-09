import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.observations import ObservationBuilder, action_mask
from lib.rolling_normalization import RollingNormalizer
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import (
    EnhancedGRPOInferenceXLSTM, GRPOInferenceE2EXLSTM, Position,
)
from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.pretrain_behavior_cloning import OfflineEpisodeDataset, distillation_loss, pretrain
from ai_trader.grpo.pretrain_behavior_cloning_fast import OfflineEpisodeDataset as FastDataset


COLUMNS = ["누적거래대금", "등락률", "현재가"]


def builder():
    return ObservationBuilder(COLUMNS, seq_len=8, rolling_window_size=3,
                              rolling_min_samples=2, max_holding_seconds=10, max_stages=5)


def write_episode(directory, suffix_multiplier=1):
    directory.mkdir()
    raw = np.column_stack((np.arange(16) * 1000 + 100, np.arange(16) * 0.01,
                           100 + np.arange(16) * 0.01)).astype(np.float32)
    raw[8:] *= suffix_multiplier
    metadata = np.array([["demo", "20260101", str(90000000 + i * 100)] for i in range(len(raw))])
    episodes = []
    for day in ("20260101", "20260102", "20260103"):
        metadata[:, 1] = day
        file_name = f"episode_{day}.npz"
        np.savez(directory / file_name, features=raw, metadata=metadata)
        episodes.append({"file_path": file_name, "length": len(raw), "date": day})
    (directory / "manifest.json").write_text(json.dumps({
        "metadata": {"feature_columns": COLUMNS, "price_unit": "krw"},
        "episodes": episodes,
    }), encoding="utf-8")
    return raw


def test_causal_bc_and_inference_share_exact_observations(tmp_path):
    spec = builder()
    raw = write_episode(tmp_path / "first")
    write_episode(tmp_path / "changed_future", suffix_multiplier=50)
    left = OfflineEpisodeDataset(tmp_path / "first", 8, step_size=1, observation_schema=spec.schema,
                                 simulate_positions=False)
    right = OfflineEpisodeDataset(tmp_path / "changed_future", 8, step_size=1,
                                  observation_schema=spec.schema, simulate_positions=False)
    live = GRPOInferenceE2EXLSTM.__new__(GRPOInferenceE2EXLSTM)
    live.observation_builder = spec
    expected = spec.build(raw[:8])
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[0], expected)
    np.testing.assert_array_equal(live.build_observation(raw[:8], feature_columns=COLUMNS), expected)
    assert FastDataset is OfflineEpisodeDataset
    assert expected.shape == (8, len(COLUMNS) + 15)
    with pytest.raises(ValueError, match="feature order"):
        live.build_observation(raw[:8], feature_columns=list(reversed(COLUMNS)))
    with pytest.raises(ValueError, match="price unit"):
        live.build_observation(raw[:8], feature_price_unit="million_krw")
    assert {ep["date"] for ep in left.episodes} == {"20260101"}


def test_statistics_use_last_available_rows_and_ordered_feature_transforms():
    spec = builder()
    raw = np.array([[10, 1, 100], [20, 2, 101], [30, 3, 102], [40, 4, 103]], dtype=float)
    transformed = raw.copy()
    transformed[:, [0, 2]] = np.log1p(transformed[:, [0, 2]])
    expected = (transformed - transformed[-3:].mean(axis=0)) / transformed[-3:].std(axis=0)
    np.testing.assert_allclose(spec.normalize(raw), expected, rtol=1e-6)
    rolling = RollingNormalizer(window_size=3, min_samples=2, feature_names=COLUMNS)
    np.testing.assert_allclose(rolling.normalize(raw), spec.normalize(raw), rtol=1e-6)
    with pytest.raises(ValueError, match="feature width"):
        rolling.normalize(raw[:, :2])


def test_stage_metadata_preserves_elapsed_time_and_five_stages():
    spec = builder()
    obs = spec.build(np.ones((8, 3)), [{"entry_price": 100, "entry_time_seconds": 1}], 101, 6)
    np.testing.assert_allclose(obs[-1, 3:6], [1, 0.01, 0.5])
    assert np.count_nonzero(obs[-1, 6:]) == 0
    np.testing.assert_array_equal(action_mask(0), [True, True, False])
    np.testing.assert_array_equal(action_mask(5, max_stages=5), [True, False, True])
    with pytest.raises(ValueError, match="after"):
        spec.build(np.ones((8, 3)), [{"entry_price": 100, "entry_time_seconds": 7}], 101, 6)


def test_schema_requires_exact_feature_order_and_version():
    spec = builder()
    assert ObservationBuilder.from_schema(spec.schema).schema == spec.schema
    with pytest.raises(ValueError, match="observation_schema"):
        ObservationBuilder.from_schema(None)
    changed = dict(spec.schema, version=1)
    with pytest.raises(ValueError, match="Incompatible"):
        ObservationBuilder.from_schema(changed)
    changed = dict(spec.schema, feature_columns=list(reversed(COLUMNS)))
    with pytest.raises(ValueError, match="Incompatible"):
        spec.validate_schema(changed)
    changed = dict(spec.schema, feature_price_unit="million_krw")
    with pytest.raises(ValueError, match="Incompatible"):
        spec.validate_schema(changed)


def make_base(action=0):
    return SimpleNamespace(observation_builder=builder(), predict=lambda *args, **kwargs: (action, 0.99))


def test_risk_uses_actual_price_not_stale_return_and_hard_time_limit():
    risk = EnhancedGRPOInferenceXLSTM(make_base(), stagnation_exit_seconds=0)
    position = Position(100, 1, 100, 0, cumulative_return=100)
    action, _, info = risk.predict(np.ones((8, 3)), position, 97, current_time_seconds=2)
    assert action == 2 and info["reason"] == "Stop Loss Hit"
    assert info["profit_rate"] == pytest.approx(-3)
    # Profitable inventory is still bounded, with elapsed time left unclipped.
    position = Position(100, 1, 100, 0)
    action, _, info = risk.predict(np.ones((8, 3)), position, 101, current_time_seconds=31)
    assert action == 2 and info["reason"] == "Maximum Holding Time"
    assert info["holding_seconds"] == 30 and info["risk_exit_all"]


def test_trailing_activation_latches_below_original_activation_threshold():
    risk = EnhancedGRPOInferenceXLSTM(make_base(), stagnation_exit_seconds=0,
                                      take_profit_rate=10, trailing_stop_activation_rate=3,
                                      max_holding_seconds=60)
    position = Position(100, 0, 100, 0)
    assert risk.predict(np.ones((8, 3)), position, 104, current_time_seconds=1)[0] == 0
    assert position.trailing_activated
    action, _, info = risk.predict(np.ones((8, 3)), position, 102, current_time_seconds=2)
    assert action == 2 and info["reason"] == "Trailing Stop Hit"


def test_stagnation_and_inventory_action_masks():
    risk = EnhancedGRPOInferenceXLSTM(make_base(action=2), stagnation_exit_seconds=2)
    assert risk.predict(np.ones((8, 3)))[0] == 0  # cannot sell flat
    position = Position(100, 0, 100, 0)
    action, _, info = risk.predict(np.ones((8, 3)), position, 100.1, current_time_seconds=3)
    assert action == 2 and info["reason"] == "Stagnation Exit"
    risk = EnhancedGRPOInferenceXLSTM(make_base(action=1), enable_auto_exit=False)
    positions = [Position(100, 0, 100, 0) for _ in range(5)]
    assert risk.predict(np.ones((8, 3)), positions, 100, current_time_seconds=1)[0] == 0


def test_default_inference_does_not_add_untrained_profit_or_stagnation_exits():
    risk = EnhancedGRPOInferenceXLSTM(make_base(), max_holding_seconds=60)
    position = Position(100, 0, 100, 0)
    assert risk.predict(np.ones((8, 3)), position, 100.1, current_time_seconds=3)[0] == 0
    assert risk.predict(np.ones((8, 3)), position, 106, current_time_seconds=4)[0] == 0
    assert risk.predict(np.ones((8, 3)), position, 102, current_time_seconds=5)[0] == 0
    assert not position.trailing_activated
    # Trained holding/stop limits remain active even though optional exits are off.
    assert risk.predict(np.ones((8, 3)), position, 97, current_time_seconds=6)[2]["reason"] == "Stop Loss Hit"


def test_inference_loads_training_stop_loss_and_allows_explicit_override(tmp_path):
    spec = builder()
    policy = GRPOPolicyE2EXLSTM(obs_dim=spec.obs_dim, cnn_channels=4,
                               rnn_hidden_dim=4, fc_hidden_dim=8, max_stages=spec.max_stages)
    path = tmp_path / "risk_policy.pt"
    torch.save({"policy_state_dict": policy.state_dict(), "observation_schema": spec.schema,
                "extra_state": {"training_config": {"stop_loss_pct": 4.0}}}, path)
    base = GRPOInferenceE2EXLSTM(path, device="cpu")
    base.predict = lambda *args, **kwargs: (0, 0.99)
    risk = EnhancedGRPOInferenceXLSTM(base)
    position = Position(100, 0, 100, 0)
    assert risk.stop_loss_rate == -4.0
    assert risk.predict(np.ones((8, 3)), position, 97, current_time_seconds=1)[0] == 0
    assert risk.predict(np.ones((8, 3)), position, 95, current_time_seconds=2)[2]["reason"] == "Stop Loss Hit"
    explicit = EnhancedGRPOInferenceXLSTM(base, stop_loss_rate=-2.0, take_profit_rate=5.0)
    assert explicit.predict(np.ones((8, 3)), Position(100, 0, 100, 0), 97,
                            current_time_seconds=1)[2]["reason"] == "Stop Loss Hit"
    assert explicit.predict(np.ones((8, 3)), Position(100, 0, 100, 0), 106,
                            current_time_seconds=1)[2]["reason"] == "Take Profit Hit"


@pytest.mark.parametrize("settings", [
    {"stop_loss_rate": np.nan}, {"stop_loss_rate": 2},
    {"take_profit_rate": np.inf}, {"trailing_stop_activation_rate": 0},
    {"trailing_stop_callback_rate": np.nan}, {"stagnation_exit_seconds": np.nan},
])
def test_invalid_inference_risk_thresholds_are_rejected(settings):
    with pytest.raises(ValueError):
        EnhancedGRPOInferenceXLSTM(make_base(), **settings)


def test_strict_policy_load_and_masked_prediction(tmp_path):
    spec = builder()
    policy = GRPOPolicyE2EXLSTM(obs_dim=spec.obs_dim, cnn_channels=4,
                               rnn_hidden_dim=4, fc_hidden_dim=8, max_stages=spec.max_stages)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 1., 20.]))
    path = tmp_path / "policy.pt"
    torch.save({"policy_state_dict": policy.state_dict(), "observation_schema": spec.schema}, path)
    inference = GRPOInferenceE2EXLSTM(path, device="cpu")
    assert inference.predict(np.ones((8, 3)))[0] == 1
    state = policy.state_dict()
    state.pop("value_head.bias")
    torch.save({"policy_state_dict": state, "observation_schema": spec.schema}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        GRPOInferenceE2EXLSTM(path, device="cpu")
    torch.save({"policy_state_dict": policy.state_dict()}, path)
    with pytest.raises(ValueError, match="observation_schema"):
        GRPOInferenceE2EXLSTM(path, device="cpu")


def test_distillation_backward_for_flat_and_full_inventory(tmp_path):
    spec = builder()
    write_episode(tmp_path / "data")
    dataset = OfflineEpisodeDataset(tmp_path / "data", 8, step_size=1, observation_schema=spec.schema)
    states = torch.stack([dataset[0], dataset[5]])
    assert states[1, -1, 3::3].sum() == 5
    teacher = GRPOPolicyE2E(obs_dim=spec.obs_dim, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8).eval()
    student = GRPOPolicyE2EXLSTM(obs_dim=spec.obs_dim, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8, max_stages=spec.max_stages)
    with torch.no_grad():
        teacher_logits, teacher_values = teacher(states)
    student_logits, student_values = student(states)
    loss = distillation_loss(teacher_logits, teacher_values, student_logits, student_values, states, 3, 2., spec.max_stages)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)


def test_tiny_distillation_checkpoint_reloads_in_shared_inference(tmp_path):
    spec = builder()
    write_episode(tmp_path / "data")
    teacher = GRPOPolicyE2E(obs_dim=spec.obs_dim, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8)
    teacher_path, output_path = tmp_path / "teacher.pt", tmp_path / "student.pt"
    torch.save({"policy_state_dict": teacher.state_dict(), "observation_schema": spec.schema,
                "extra_state": {"date_splits": {"train": ["20260101"], "validation": ["20260102"]}}}, teacher_path)
    pretrain(str(teacher_path), str(tmp_path / "data"), str(output_path), epochs=1,
             batch_size=2, step_size=8, device="cpu")
    inference = GRPOInferenceE2EXLSTM(output_path, device="cpu")
    assert inference.observation_schema == spec.schema
    saved = torch.load(output_path, weights_only=True)
    assert saved["extra_state"]["date_splits"] == {"train": ["20260101"], "validation": ["20260102"],
                                                 "test": ["20260103"]}


def test_distillation_rejects_unknown_or_contaminated_teacher_dates(tmp_path):
    spec = builder()
    write_episode(tmp_path / "data")
    path = tmp_path / "teacher.pt"
    checkpoint = {"policy_state_dict": {}, "observation_schema": spec.schema}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="provenance"):
        pretrain(str(path), str(tmp_path / "data"), str(tmp_path / "result.pt"), device="cpu")
    checkpoint["extra_state"] = {"date_splits": {"train": ["20260102"]}}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="training dates contaminate"):
        pretrain(str(path), str(tmp_path / "data"), str(tmp_path / "result.pt"), device="cpu")
    checkpoint["extra_state"] = {"date_splits": {"train": ["20260101"], "validation": ["20260103"]}}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="validation dates contaminate"):
        pretrain(str(path), str(tmp_path / "data"), str(tmp_path / "result.pt"), device="cpu")
