"""Fixed, server-owned method choices; isolate legacy imports in a subprocess."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'v0.81': {'label':'v0.81 · 原版流程（无 Mask 修复）','root':ROOT.parent/'v0.81','engine':'legacy',
               'description':'原版 G-Cut 和剪枝；第 7 步直接使用合并 mask。'},
    'v0.82': {'label':'v0.82 · GSDT / 三维合并（无 Mask 修复）','root':ROOT.parent/'v0.82','engine':'legacy',
               'description':'使用 v0.82 的 APP2 GSDT、候选证据过滤、三维合并与原版剪枝；不补全 mask。'},
    'v0.82a': {'label':'v0.82a · gamma / GPU 前处理（无 Mask 修复）','root':ROOT.parent/'v0.82a','engine':'legacy',
                'description':'原网格 gamma=0.5 和 GPU 1 µm 重采样；其余沿用该版本原有 G-Cut 与剪枝。已有 1 µm 输入会跳过这些前处理。'},
    'v0.83': {'description':'原 XY MIP 引导的 Mask 修复及原版后处理。','label': 'v0.83 · 原 XY MIP', 'root': ROOT.parent / 'v0.83'},
    'v0.83a': {'description':'三方向、重叠、多厚度 MIP；根部测地线补全已停用，G-Cut 短末梢阈值 7.75 µm。','label': 'v0.83a · 三方向、重叠、多厚度 MIP', 'root': ROOT},
}
DEFAULT_METHOD = 'v0.83a'


def method_root(version):
    if not isinstance(version, str) or version not in METHODS:
        raise ValueError('请选择列表中的算法流程版本')
    root = METHODS[version]['root']
    entry='run_pipeline.py' if METHODS[version].get('engine')=='legacy' else 'web_console/runner.py'
    if not (root / entry).is_file():
        raise ValueError(f'{version} 代码目录不可用')
    return root


def stage_names(version, default):
    names=list(default)
    if METHODS[version].get('engine')=='legacy':
        names[6]='使用合并 mask（不修复）'
        names[11]='版本记录（不替换历史结果）'
    return names


def runner_entry(version):
    root=method_root(version)
    return (ROOT if METHODS[version].get('engine')=='legacy' else root)/'web_console/runner.py'


def preflight_request(req):
    root = method_root(req['method_version'])
    if root == ROOT or METHODS[req['method_version']].get('engine')=='legacy':
        from web_console.runner import preflight
        return preflight(req)
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), req['method_version']],
                            input=json.dumps(req), text=True, capture_output=True, timeout=300,
                            cwd=root)
    try:
        response = json.loads(result.stdout)
    except ValueError:
        raise ValueError(f"{req['method_version']} 输入检查进程失败，请检查该版本环境")
    if result.returncode or not response.get('ok'):
        raise ValueError(response.get('error', '输入检查失败'))
    return response['plan']


if __name__ == '__main__':
    try:
        root = method_root(sys.argv[1])
        sys.path.insert(0, str(root))
        with contextlib.redirect_stdout(sys.stderr):
            from web_console.runner import preflight
            plan = preflight(json.load(sys.stdin))
        print(json.dumps({'ok': True, 'plan': plan}))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        sys.exit(1)
