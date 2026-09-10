"""Execute the notebook's real TAR preparation helpers without Colab or CUDA."""
import ast
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb'


def helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code = next(''.join(cell['source']) for cell in notebook['cells'] if cell.get('id') == 'data-functions')
    scope = {}
    exec(compile(code, str(NOTEBOOK), 'exec'), scope)
    return scope


def make_tar(path, prefix='', *, extras=(), omit_npz=False, byte_delta=0):
    payload = io.BytesIO()
    np.savez_compressed(payload, features=np.ones((2, 3), dtype=np.float32))
    npz = payload.getvalue()
    manifest = {'metadata': {'schema_version': 2}, 'episodes': [
        {'file_path': 'episode.npz', 'date': '20260101', 'length': 2, 'bytes': len(npz) + byte_delta}]}
    raw_manifest = json.dumps(manifest, indent=2).encode()
    members = [(prefix + 'manifest.json', raw_manifest), (prefix + 'progress.json', b'{}')]
    if not omit_npz:
        members.append((prefix + 'episode.npz', npz))
    with tarfile.open(path, 'w') as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        for info, content in extras:
            archive.addfile(info, io.BytesIO(content) if content is not None else None)
    return raw_manifest, npz


@pytest.mark.parametrize('prefix', ['', './', 'extracted_episodes_v2/'])
def test_tar_roundtrip_and_reuse_never_reads_drive_npzs(tmp_path, monkeypatch, prefix):
    archive = tmp_path / 'drive.tar'
    original_manifest, original_npz = make_tar(archive, prefix)
    prepare = helpers()['prepare_episode_tar']
    dataset, manifest = prepare(archive, tmp_path / 'cache', reserve_bytes=0)
    assert manifest == original_manifest  # Resume manifest fingerprint stays identical.
    assert (dataset / 'episode.npz').read_bytes() == original_npz
    assert not (dataset / 'progress.json').exists()
    with np.load(dataset / 'episode.npz', allow_pickle=False) as data:
        assert data['features'].shape == (2, 3)
    original_open = Path.open

    def forbid_drive_reads(path, *args, **kwargs):
        assert path != archive, 'The completed archive must not be read from Drive again'
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', forbid_drive_reads)
    assert prepare(archive, tmp_path / 'cache', reserve_bytes=0) == (dataset, manifest)
    # Rebuild a missing local NPZ from the cached TAR, also without Drive reads.
    (dataset / 'episode.npz').unlink()
    assert prepare(archive, tmp_path / 'cache', reserve_bytes=0) == (dataset, manifest)
    assert (dataset / 'episode.npz').read_bytes() == original_npz


def test_interrupted_extraction_does_not_publish_manifest_and_can_resume(tmp_path, monkeypatch):
    archive = tmp_path / 'drive.tar'
    original_manifest, npz = make_tar(archive)
    scope = helpers()
    original_copy = scope['shutil'].copyfileobj

    def interrupted(source, target, **kwargs):
        target.write(source.read(10))
        raise RuntimeError('interrupted')

    monkeypatch.setattr(scope['shutil'], 'copyfileobj', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        scope['prepare_episode_tar'](archive, tmp_path / 'cache', reserve_bytes=0)
    assert not list((tmp_path / 'cache').rglob('manifest.json'))
    assert not list((tmp_path / 'cache').rglob('.tar_ready.json'))
    monkeypatch.setattr(scope['shutil'], 'copyfileobj', original_copy)
    dataset, manifest = scope['prepare_episode_tar'](archive, tmp_path / 'cache', reserve_bytes=0)
    assert manifest == original_manifest and (dataset / 'episode.npz').read_bytes() == npz


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/escape', 'file'),
                                    ('C:/escape', 'file'), ('dir\\escape', 'file'),
                                    ('link', 'symlink'), ('link', 'hardlink'),
                                    ('episode.npz', 'file')])
def test_unsafe_archive_rejected_before_dataset_publication(tmp_path, name, kind):
    member = tarfile.TarInfo(name)
    member.size = 1 if kind == 'file' else 0
    if kind != 'file':
        member.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
        member.linkname = 'episode.npz'
    archive = tmp_path / 'drive.tar'
    make_tar(archive, extras=[(member, b'x' if kind == 'file' else None)])
    with pytest.raises(ValueError):
        helpers()['prepare_episode_tar'](archive, tmp_path / 'cache', reserve_bytes=0)
    assert not list((tmp_path / 'cache').rglob('manifest.json'))
    assert not list((tmp_path / 'cache').rglob('episode.npz'))


@pytest.mark.parametrize('options', [{'omit_npz': True}, {'byte_delta': 1}])
def test_missing_or_wrong_size_npz_rejected(tmp_path, options):
    archive = tmp_path / 'drive.tar'
    make_tar(archive, **options)
    with pytest.raises(ValueError, match='NPZ|mismatch'):
        helpers()['prepare_episode_tar'](archive, tmp_path / 'cache', reserve_bytes=0)
    assert not list((tmp_path / 'cache').rglob('.tar_ready.json'))


def test_insufficient_disk_space_fails_before_copy(tmp_path, monkeypatch):
    archive = tmp_path / 'drive.tar'
    make_tar(archive)
    scope = helpers()
    monkeypatch.setattr(scope['shutil'], 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='공간 부족'):
        scope['prepare_episode_tar'](archive, tmp_path / 'cache')
    assert not list((tmp_path / 'cache').glob('*.tar*'))


def test_interrupted_tar_copy_reclaims_its_partial_file_on_tight_disk(tmp_path, monkeypatch):
    archive, cache = tmp_path / 'drive.tar', tmp_path / 'cache'
    manifest, _ = make_tar(archive)
    payload = archive.read_bytes()
    scope, original_open = helpers(), Path.open

    class InterruptedRead(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise RuntimeError('copy interrupted')
            return super().read(size)

    def interrupt_archive(path, *args, **kwargs):
        return InterruptedRead(payload) if path == archive else original_open(path, *args, **kwargs)

    # Enough for TAR+extraction, but not for another full copy reservation on top of .part.
    def disk_usage(path):
        used = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        return SimpleNamespace(free=len(payload) * 5 // 2 - used)

    monkeypatch.setattr(scope['shutil'], 'disk_usage', disk_usage)
    monkeypatch.setattr(Path, 'open', interrupt_archive)
    with pytest.raises(RuntimeError, match='copy interrupted'):
        scope['prepare_episode_tar'](archive, cache, reserve_bytes=0)
    assert list(cache.glob('*.tar.part'))
    assert not list(cache.glob('*.copy.json'))
    monkeypatch.setattr(Path, 'open', original_open)
    dataset, result = scope['prepare_episode_tar'](archive, cache, reserve_bytes=0)
    assert result == manifest and (dataset / '.tar_ready.json').is_file()


def test_notebook_remains_valid_and_exposes_required_downstream_variables():
    import nbformat
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    code = next(cell.source for cell in notebook.cells if cell.id == 'data')
    assert all(name in code for name in ('LOCAL_DATA_DIR', 'manifest_bytes', 'MANIFEST_SHA256', 'manifest'))
    assert 'DRIVE_TAR_PATH' in code and 'source.stat' not in code
