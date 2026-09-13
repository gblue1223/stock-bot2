"""Exercise the dedicated notebook without Drive, CUDA, or starting training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trader.grpo.entry_pattern import DEFAULT_ENTRY_PATTERN
from ai_trader.grpo.train_xlstm import TrainingConfig
from lib.observations import ObservationBuilder


PATH = Path(__file__).parents[1] / 'ai_trader/grpo/colab_train_xlstm_entry_pattern_2048.ipynb'


def sources():
    return {cell['id']: ''.join(cell['source']) for cell in json.loads(PATH.read_text(encoding='utf-8'))['cells']}


def scope():
    result = {'GiB': 1024 ** 3, 'ObservationBuilder': ObservationBuilder}
    for name in ('profile-functions', 'entry-pattern-check-functions'):
        exec(compile(sources()[name], name, 'exec'), result)
    return result


def dataset():
    return {'metadata': {'entry_pattern_config': dict(DEFAULT_ENTRY_PATTERN), 'expected_features': 29,
                         'extraction': {'buy_notional_source': 'signed_volume_positive_buy'}},
            'episodes': [{'length': 10000, 'entry_signal_ranges': [[2047, 2200]]}]}


def config():
    return {'seq_len': 2048, 'features': 29, 'entry_pattern_config': dict(DEFAULT_ENTRY_PATTERN),
            'execution_action_mask': True}


def test_default_config_executes_and_keeps_requested_contract():
    ns = scope()
    ns.update(Path=Path, json=json, TrainingConfig=TrainingConfig,
              torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (38 * 1024**3, 40 * 1024**3))),
              VRAM_GIB=40, CPU_COUNT=12, available_ram_gib=lambda: 50,
              LOCAL_DATA_DIR=Path('/unused/data'), MANIFEST_SHA256='test', manifest=dataset())
    exec(compile(sources()['config'], 'config', 'exec'), ns)
    assert ns['MODE'] == 'new' and not ns['LOAD_POLICY']
    assert ns['CONFIG']['seq_len'] == 2048
    assert ns['CONFIG']['entry_pattern_config'] == DEFAULT_ENTRY_PATTERN
    assert ns['CONFIG']['decision_interval_seconds'] == 1
    assert ns['ENTRY_DATA_SUMMARY']['qualifying_episodes'] == 1


@pytest.mark.parametrize('change', [{'seq_len': 1024}, {'entry_pattern_config': None},
    {'execution_action_mask': False}, {'entry_pattern_config': {**DEFAULT_ENTRY_PATTERN, 'min_price_return': .02}}])
def test_rejects_old_or_different_checkpoint_contract(change):
    with pytest.raises(ValueError):
        scope()['validate_entry_pattern_dataset']({**config(), **change}, dataset())


@pytest.mark.parametrize('broken', ['legacy', 'wrong_source', 'no_entries', 'too_early', 'different_threshold'])
def test_rejects_incompatible_extracted_data(broken):
    data = copy.deepcopy(dataset())
    if broken == 'legacy':
        data['metadata'].pop('entry_pattern_config')
    elif broken == 'wrong_source':
        data['metadata']['extraction']['buy_notional_source'] = 'total_turnover'
    elif broken == 'no_entries':
        data['episodes'] = []
    elif broken == 'too_early':
        data['episodes'][0]['entry_signal_ranges'] = [[2, 1000]]
    else:
        data['metadata']['entry_pattern_config']['min_buy_notional_krw'] = 2e9
    with pytest.raises(ValueError):
        scope()['validate_entry_pattern_dataset'](config(), data)
