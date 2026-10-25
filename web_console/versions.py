"""Fixed, server-owned method choices; isolate legacy imports in a subprocess."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'v0.83': {'label': 'v0.83 · 原 XY MIP', 'root': ROOT.parent / 'v0.83'},
    'v0.83a': {'label': 'v0.83a · 三方向、重叠、多厚度 MIP', 'root': ROOT},
}
DEFAULT_METHOD = 'v0.83a'


def method_root(version):
    if not isinstance(version, str) or version not in METHODS:
        raise ValueError('请选择 v0.83 或 v0.83a')
    root = METHODS[version]['root']
    if not (root / 'web_console/runner.py').is_file():
        raise ValueError(f'{version} 代码目录不可用')
    return root


def preflight_request(req):
    root = method_root(req['method_version'])
    if root == ROOT:
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
