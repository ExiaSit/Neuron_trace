"""Sample-local, content-addressed cache of mask-independent image evidence.

Only stages in an explicit RECONNECT_WORK directory share data. Content hashes,
configuration and implementation hashes prevent reuse after input/code changes.
An atomic metadata file commits each generation; incomplete files are ignored.
The ordinary mask-step cleanup also removes this temporary cache.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from functools import lru_cache
import numpy as np


def key_for(arrays, configuration):
    h=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode())
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str((a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    h.update(implementation_digest())
    return h.hexdigest()


@lru_cache(maxsize=1)
def implementation_digest():
    # Stage workers are separate processes. Hash loaded implementation once
    # per process; a later stage/run automatically gets the updated key.
    h=hashlib.sha256()
    for name in ('image_evidence_cache.py','mip_backmap.py','multiscale_mip.py','multiscale_mip_cuda.py'):
        p=Path(__file__).with_name(name)
        if p.exists():h.update(p.read_bytes())
    h.update((Path(__file__).parents[2]/'utils/gpu_ndimage.py').read_bytes())
    return h.digest()


def cached_arrays(name, arrays, configuration, compute):
    work=os.environ.get('RECONNECT_WORK')
    if not work or os.environ.get('RECONNECT_EVIDENCE_CACHE','1')=='0':return compute()
    started=time.perf_counter();key=key_for(arrays,configuration)
    root=Path(work)/'image_evidence_cache';root.mkdir(parents=True,exist_ok=True)
    directory=root/(name+'-'+key);meta=directory/'complete.json'
    try:
        info=json.loads(meta.read_text());assert info['key']==key
        values=[]
        for entry in info['arrays']:
            a=np.load(directory/entry['file'],mmap_mode='r',allow_pickle=False)
            if list(a.shape)!=entry['shape'] or a.dtype.str!=entry['dtype']:raise ValueError('Invalid cached array')
            values.append(a)
        result=(*values,info['report']);hit=True
    except (OSError,ValueError,KeyError,AssertionError,EOFError):
        result=compute();directory.mkdir(exist_ok=True);entries=[]
        for i,a in enumerate(result[:-1]):
            target=directory/f'{i}.npy'
            with tempfile.NamedTemporaryFile(dir=directory,suffix='.npy',delete=False) as f:
                tmp=Path(f.name)
                try:np.save(f,a,allow_pickle=False)
                except BaseException:tmp.unlink(missing_ok=True);raise
            os.replace(tmp,target)
            entries.append(dict(file=target.name,shape=list(a.shape),dtype=a.dtype.str))
        info=dict(key=key,arrays=entries,report=result[-1])
        with tempfile.NamedTemporaryFile(mode='w',dir=directory,suffix='.json',delete=False) as f:
            json.dump(info,f);tmp=Path(f.name)
        os.replace(tmp,meta);hit=False
    print(f'[image evidence cache] {name}: {"hit" if hit else "computed"}, {time.perf_counter()-started:.3f}s',flush=True)
    return result


def raw_contrast(raw):
    from utils.gpu_ndimage import ndi
    def compute():
        f=raw.astype('float32')
        return ndi.gaussian_filter(f,(0,.6,.6))-ndi.gaussian_filter(f,(0,4,4)),{}
    return cached_arrays('raw_contrast',[raw],dict(backend=os.environ.get('RECONNECT_COMPUTE_BACKEND','cpu'),
                         minimum=os.environ.get('RECONNECT_GPU_MIN_VOXELS','262144')),compute)[0]
