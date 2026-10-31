"""Keep a byte-exact, independent original mask beside repaired outputs."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def same_file(a, b):
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def preserve_original_mask(spec):
    """Publish once; reject input/output aliasing and changed originals on resume."""
    cid = str(spec['id'])
    if not cid.isdigit():
        raise ValueError('Expected numeric ID')
    source = Path(spec['mask'])
    root = Path(spec['config']['output_root'])
    saved = root / 'mask_original' / f'{cid}.tif'
    repaired = root / 'mask' / f'{cid}.tif'
    if same_file(source, repaired) or same_file(saved, repaired):
        raise ValueError('原始 mask 与修复输出不能指向同一文件；请使用独立的 output_root')
    if not source.is_file() or not source.stat().st_size:
        raise ValueError(f'原始 mask 不存在或为空：{source}')
    expected = sha256(source)
    if saved.exists():
        if sha256(saved) != expected:
            raise ValueError(f'原始 mask 与已保存副本不同，拒绝覆盖：{saved}')
        return saved
    saved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=saved.parent, suffix='.writing', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected or sha256(source) != expected:
            raise ValueError('复制期间原始 mask 发生变化，未发布副本')
        # Publish atomically without replacing an existing original, even if
        # another worker saved it concurrently. This never links the source.
        try:
            os.link(temporary, saved)
        except FileExistsError:
            if sha256(saved) != expected:
                raise ValueError(f'原始 mask 副本已存在且内容不同：{saved}')
    finally:
        temporary.unlink(missing_ok=True)
    return saved
