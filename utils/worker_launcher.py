import os, shutil, tempfile, time
from pathlib import Path
from utils import case_worker as worker

def atomic_copy(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source,name)
        os.replace(name,destination)
    finally:
        if Path(name).exists():Path(name).unlink()

def local_app2(native,img_file,marker_file,out_swc_file,timeout_sec,native_log=None,temp_root='/tmp/neuron-app2-v083'):
    out=Path(out_swc_file)
    if out.exists():
        worker.check_swc(out)
        return
    scratch=Path(temp_root);scratch.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='trace-',dir=scratch) as directory:
        local=Path(directory)/'native.swc';log=Path(directory)/'native.log'
        try:
            native(img_file,marker_file,str(local),timeout_sec,native_log=log)
            worker.check_swc(local)
            atomic_copy(local,out)
        finally:
            if native_log and log.exists():atomic_copy(log,native_log)
            if native_log:
                worker.save_json(Path(native_log).with_name('app2-io.json'),dict(
                    revision='local-scratch-1',scratch_parent=str(scratch),seconds=time.monotonic()-start,
                    success=out.is_file(),tracing_parameters_changed=False))
