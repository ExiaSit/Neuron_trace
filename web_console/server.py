#!/usr/bin/env python3
"""LAN console. Fixed pipeline commands, authenticated API, job-scoped file serving."""
import argparse
import hmac
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from web_console.runner import STAGES, ROOT, parse_request, preflight
from utils.case_worker import save_json

STATIC=Path(__file__).parent/'static'
READABLE={'.png','.jpg','.jpeg','.webp','.swc','.tif','.tiff','.json','.log','.txt'}
HISTORY=Path('/media/pzy/U-Disk2/human_reconstruction/v0.83-ring-guard-20261016')
BROWSE_ROOTS=[Path('/data'),Path('/mnt'),Path('/media'),Path.home()]


def browse_directory(value='',kind='directory',query='',offset=0,roots=None):
    """Authenticated picker metadata only; never read or serve input contents."""
    if kind not in ('directory','file'):raise ValueError('选择类型无效')
    if not isinstance(offset,int) or not 0<=offset<=100000:raise ValueError('分页位置无效')
    if len(value)>4096 or len(query)>256:raise ValueError('路径或搜索文本过长')
    roots=[p.resolve() for p in (BROWSE_ROOTS if roots is None else roots) if p.is_dir()]
    if not roots:raise ValueError('没有可浏览的数据目录')
    def allowed(p):return any(p==r or r in p.parents for r in roots)
    raw=Path(value) if value else roots[0]
    if not raw.is_absolute():raise ValueError('请输入服务器上的绝对路径')
    path=raw.resolve()
    if not allowed(path):raise ValueError('请从提供的数据目录中选择文件或文件夹')
    selected=None
    if path.is_file():
        if kind!='file' or path.suffix.lower()!='.csv':raise ValueError('元数据请选择 CSV 文件')
        selected=str(path);path=path.parent
    if not path.is_dir():raise ValueError('目录不存在，请选择上级目录或常用位置')
    entries=[];query=query.casefold()
    try:
        with os.scandir(path) as scan:
            for entry in scan:
                if entry.name.startswith('.') or query not in entry.name.casefold():continue
                try:
                    link=entry.is_symlink()
                    # Avoid per-file stat on large image directories.
                    directory=entry.is_dir(follow_symlinks=False)
                    if not directory and not link and (kind=='directory' or Path(entry.name).suffix.lower()!='.csv'):continue
                    resolved=Path(entry.path).resolve() if link else Path(entry.path)
                    if not allowed(resolved):continue
                    if link:directory=entry.is_dir()
                    if not directory and (kind=='directory' or resolved.suffix.lower()!='.csv' or not entry.is_file()):continue
                    entries.append(dict(name=entry.name,path=str(resolved),directory=directory,symlink=link))
                except (OSError,RuntimeError):continue
    except PermissionError:raise ValueError('当前服务用户没有权限读取此目录')
    entries.sort(key=lambda e:(not e['directory'],e['name'].casefold(),e['name']))
    total=len(entries);limit=100
    return dict(path=str(path),parent=str(path.parent) if allowed(path.parent) and path.parent!=path else None,
                roots=list(map(str,roots)),entries=entries[offset:offset+limit],total=total,offset=offset,
                limit=limit,has_more=offset+limit<total,selected=selected,kind=kind)


def contained(root,relative):
    path=(root/relative).resolve();base=root.resolve()
    if path==base or base not in path.parents:raise ValueError('无效文件路径')
    return path


def tail(path,limit=48000):
    if not path.is_file():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));return f.read(limit).decode('utf-8',errors='replace')


def artifacts(root,cid):
    result=[]
    # Scan only known figure/output directories, never raw source directories.
    for folder in ['mask_step_mips','gcut_audits','comparison_sources','preparation_mips','app2_mips','mask_mips','gcut_mips','pruning_mips','comparisons','pruning_swcs',
                   'trace_app2/down_sampled_swcs_app2','gcut_output/selected_for_pruning',
                   'gcut_output/selected_resampled_step1','regression_guard']:
        directory=root/folder
        if not directory.is_dir():continue
        for p in directory.iterdir():
            if p.suffix in {'.png','.swc','.json'} and re.match(rf'^(?:image_)?{re.escape(cid)}(?:[_.]|$)',p.name) and p.is_file():
                if folder=='comparisons' and p.name==f'{cid}.png' and (directory/f'{cid}_diagnostics.png').is_file():continue
                result.append(dict(path=str(p.relative_to(root)),name=p.name,group=folder,
                                   image=p.suffix=='.png',size=p.stat().st_size,mtime=p.stat().st_mtime_ns))
    for item in result:
        if not item['image']:continue
        try:
            if item['group']=='mask_step_mips':
                info=json.loads((root/item['path']).with_suffix('.json').read_text())
                note='复算验证' if info['provenance'].startswith('replayed') else '保留的历史快照' if info['provenance'].startswith('retained') else '合并变化；中间快照不可用' if info['provenance'].startswith('combined') else '运行时记录'
                item['description']=f"新增 {info['added_voxels']:,} 体素 · 删除 {info['removed_voxels']:,} 体素 · {note}"
            elif item['group']=='gcut_audits':
                info=json.loads((root/item['path']).with_suffix('.json').read_text())
                note=' · 根据保留输入复算' if info.get('provenance','').startswith('replayed') else ''
                item['description']=f"短末梢规则删除 {len(info['removed_ids'])} 个节点；已排除胞体收缩与后续 G-Cut 分离"+note
            elif item['group']=='comparisons':
                info=json.loads((root/'comparison_sources'/f'{cid}_app2.json').read_text())
                item['description']=('旧版 APP2 来源：'+info['path']+' · 坐标转换：'+info['transform']) if info['status']=='present' else '配置目录中没有匹配此 ID 的旧版 APP2'
        except (OSError,ValueError,KeyError,TypeError):
            pass  # The image can appear just before its atomic sidecar is published.
    return result


class Manager:
    def __init__(self,root):
        import fcntl
        self.root=root.resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.instance_lock=(self.root/'server.lock').open('a')
        fcntl.flock(self.instance_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.jobs=self.root/'jobs';self.jobs.mkdir(exist_ok=True)
        self.lock=threading.RLock();self.active=None;self.process=None
        self.token_path=self.root/'control-token'
        if not self.token_path.exists():
            fd=os.open(self.token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
        os.chmod(self.token_path,0o600);self.token=self.token_path.read_text().strip()
        for job in self.jobs.glob('*'):
            if (job/'job.json').exists():
                p=job/'progress.json'
                state=json.loads(p.read_text()) if p.exists() else {}
                if state.get('status') in ('running','pausing','queued'):
                    state.update(status='interrupted',error='服务重启，任务已中断；可检查后续跑');save_json(p,state)

    def directory(self,jid):
        if not re.fullmatch(r'[a-f0-9]{16}',jid):raise ValueError('任务 ID 无效')
        p=self.jobs/jid
        if not (p/'job.json').is_file():raise ValueError('任务不存在')
        return p

    def summary(self,job):
        record=json.loads((job/'job.json').read_text());p=job/'progress.json'
        progress=json.loads(p.read_text()) if p.exists() else {'status':'queued','completed':{}}
        return dict(id=job.name,name=record['request']['name'],request=record['request'],
                    created_at=record['created_at'],output_root=str(job/'reconstruction'),
                    prepared_root=str(job/'prepared'),progress=progress,active=self.active==job.name)

    def all(self):
        return sorted([self.summary(p.parent) for p in self.jobs.glob('*/job.json')],key=lambda x:x['created_at'],reverse=True)

    def create(self,body):
        req=parse_request(body)
        with self.lock:
            if self.active:raise ValueError('已有运行中的任务，请等待完成或暂停后再启动')
            plan=preflight(req)
            jid=uuid.uuid4().hex[:16];job=self.jobs/jid;job.mkdir()
            save_json(job/'job.json',dict(request=req,plan=plan,created_at=time.time()))
            save_json(job/'progress.json',dict(status='queued',completed={},failed={}))
            self.start(jid);return self.summary(job)

    def start(self,jid):
        with self.lock:
            if self.active:raise ValueError('已有运行中的任务')
            job=self.directory(jid);state=self.summary(job)['progress']
            if state.get('status')=='complete':raise ValueError('该实验已完成；改变范围或参数请创建新实验')
            pause=job/'pause.request'
            if pause.exists():pause.unlink()
            state.update(status='running');save_json(job/'progress.json',state)
            log=(job/'worker.log').open('a')
            try:
                self.process=subprocess.Popen([sys.executable,str(Path(__file__).parent/'runner.py'),'--job',str(job)],
                                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                              env=dict(os.environ,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            finally:log.close()
            self.active=jid
            threading.Thread(target=self.watch,args=(jid,self.process),daemon=True).start()

    def watch(self,jid,proc):
        code=proc.wait()
        with self.lock:
            job=self.directory(jid);p=job/'progress.json';state=json.loads(p.read_text())
            if state.get('status') in ('running','queued','pausing'):
                state.update(status='failed' if code else 'complete',error=f'任务进程退出（{code}），请检查日志' if code else '')
                save_json(p,state)
            if self.active==jid:self.active=None;self.process=None

    def pause(self,jid):
        with self.lock:
            if self.active!=jid:raise ValueError('该任务未在运行')
            job=self.directory(jid);(job/'pause.request').touch()
            # The worker owns progress.json; do not race its atomic checkpoints.

    def stop(self,jid):
        import psutil
        with self.lock:
            if self.active!=jid:raise ValueError('该任务未在运行')
            job=self.directory(jid);proc=self.process
            try:
                parent=psutil.Process(proc.pid);children=parent.children(recursive=True)
                parent.suspend()
                for child in children:
                    try:child.suspend()
                    except psutil.Error:pass
                for child in reversed(children):
                    try:child.kill()
                    except psutil.Error:pass
                parent.kill()
                psutil.wait_procs(children,timeout=3)
            except psutil.NoSuchProcess:pass
            proc.wait()
            p=job/'progress.json';state=json.loads(p.read_text())
            state.update(status='interrupted',error='用户立即停止；未完成步骤将在续跑时重做')
            save_json(p,state)
            self.active=None;self.process=None


class Handler(BaseHTTPRequestHandler):
    server_version='NeuronConsole/1.0'
    def json(self,value,code=200):
        data=json.dumps(value,ensure_ascii=False,default=str).encode()
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def authorized(self,query):
        token=self.headers.get('X-Control-Token','')
        # Cookie is HttpOnly; file/image requests cannot attach custom headers.
        if not token:
            from http.cookies import SimpleCookie
            try:token=SimpleCookie(self.headers.get('Cookie','')).get('neuron_token').value
            except (AttributeError,ValueError):token=''
        return hmac.compare_digest(token,self.server.manager.token)

    def file(self,path,download=False):
        if not path.is_file():raise FileNotFoundError('文件不存在')
        self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(path.stat().st_size));self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-cache')
        if download:self.send_header('Content-Disposition','attachment')
        self.end_headers()
        with path.open('rb') as f:
            while True:
                block=f.read(256*1024)
                if not block:break
                self.wfile.write(block)

    def do_GET(self):
        try:self.get()
        except (BrokenPipeError,ConnectionResetError):pass
        except FileNotFoundError as exc:self.json({'error':str(exc)},404)
        except Exception as exc:self.json({'error':str(exc)},400)

    def get(self):
        parsed=urlsplit(self.path);path=unquote(parsed.path);q=parse_qs(parsed.query);m=self.server.manager
        if path in ('/','/index.html','/app.js','/picker.js','/style.css'):
            return self.file(STATIC/('index.html' if path=='/' else path.lstrip('/')))
        if path=='/api/bootstrap':
            cfg=json.loads((ROOT/'config.example.json').read_text())
            defaults=dict(metadata_file=cfg['metadata_file'],v3d_root=cfg['v3d_root'])
            defaults_path=ROOT/'web_console/input_defaults.json'
            if defaults_path.is_file():defaults.update(json.loads(defaults_path.read_text()))
            return self.json(dict(stages=STAGES,input_defaults=defaults,metadata_file=cfg['metadata_file'],v3d_root=cfg['v3d_root'],
                                  revision=cfg['method_revision'],history_available=HISTORY.is_dir(),
                                  token_file=str(m.token_path),job_root=str(m.jobs)))
        if not self.authorized(q):return self.json({'error':'请先输入控制口令连接工作台'},401)
        if path=='/api/browse':
            return self.json(browse_directory(q.get('path',[''])[0],q.get('kind',['directory'])[0],
                                              q.get('q',[''])[0],int(q.get('offset',['0'])[0])))
        if path=='/api/jobs':return self.json(m.all())
        if path=='/api/job':
            job=m.directory(q.get('id',[''])[0]);data=m.summary(job)
            cid=q.get('cell',[data['request']['ids'][0]])[0]
            if cid not in data['request']['ids']:raise ValueError('样本不属于本任务')
            data['artifacts']=artifacts(job/'reconstruction',cid)
            data['pause_requested']=(job/'pause.request').exists() and data['active']
            current=data['progress'].get('current');log=q.get('log',[''])[0]
            if log and re.fullmatch(r'\d{1,12}-\d{2}',log):logpath=job/'logs'/f'{log}.log'
            elif current:logpath=job/'logs'/f"{current['id']}-{current['stage']:02d}.log"
            else:logpath=job/'worker.log'
            data['log']=tail(logpath);data['log_name']=str(logpath.relative_to(job))
            return self.json(data)
        if path=='/api/history':
            cid=q.get('cell',['16984'])[0]
            if not re.fullmatch(r'\d{1,12}',cid):raise ValueError('ID 无效')
            return self.json(dict(id=cid,artifacts=artifacts(HISTORY,cid),source=str(HISTORY)))
        if path=='/api/file':
            relative=q.get('path',[''])[0]
            if q.get('history')==['1']:
                root=HISTORY
                if relative not in {a['path'] for a in artifacts(root,q.get('cell',[''])[0])}:raise ValueError('未登记的结果文件')
            else:
                job=m.directory(q.get('id',[''])[0]);root=job/'reconstruction'
            p=contained(root,relative)
            if p.suffix.lower() not in READABLE:raise ValueError('不支持的文件类型')
            return self.file(p,p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'})
        self.json({'error':'接口不存在'},404)

    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):return self.json({'error':'不接受跨站请求'},403)
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.json({'error':'需要 JSON 请求'},415)
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=32768:raise ValueError('请求大小无效')
            body=json.loads(self.rfile.read(size));path=urlsplit(self.path).path;m=self.server.manager
            if path=='/api/login':
                if not hmac.compare_digest(str(body.get('token','')),m.token):return self.json({'error':'控制口令不正确'},401)
                data=b'{"ok":true}';self.send_response(200)
                self.send_header('Set-Cookie',f'neuron_token={m.token}; Path=/; HttpOnly; SameSite=Strict')
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
                self.end_headers();self.wfile.write(data);return
            if not self.authorized({}):return self.json({'error':'需要控制口令'},401)
            if path=='/api/check':
                req=parse_request(body);plan=preflight(req)
                return self.json(dict(ok=True,ids=req['ids'],required=plan['required'],stages=STAGES[req['start']-1:req['stop']],
                                      message=f"{len(req['ids'])} 个样本输入检查通过"))
            if path=='/api/jobs':return self.json(m.create(body),201)
            jid=body.get('id','')
            if path=='/api/pause':m.pause(jid)
            elif path=='/api/stop':m.stop(jid)
            elif path=='/api/resume':m.start(jid)
            else:return self.json({'error':'接口不存在'},404)
            self.json({'ok':True})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as exc:self.json({'error':str(exc)},400)

    def log_message(self,fmt,*args):
        # Do not log credentials or request bodies.
        print(f'{self.address_string()} {fmt % args}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8766)
    ap.add_argument('--state-root',type=Path,default=ROOT.parent/'results/v083-console')
    args=ap.parse_args();manager=Manager(args.state_root)
    server=ThreadingHTTPServer((args.host,args.port),Handler);server.manager=manager
    print(f'Neuron console: http://{args.host}:{args.port}/ ; control token file: {manager.token_path}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
