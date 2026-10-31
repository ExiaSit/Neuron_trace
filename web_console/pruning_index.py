"""Read per-cell pruning evidence for a single registered experiment."""
import json
from collections import Counter
from pathlib import Path
import re


def pruning_index(root, ids):
    root = Path(root).resolve()
    cells, rules = [], {}
    for cid in ids:
        if not re.fullmatch(r'\d{1,12}', str(cid)):
            raise ValueError('ID 无效')
        cid = str(cid)
        counts, names, source, errors = None, {}, None, []
        for relative, kind in [(f'logs/{cid}/pruning-audit.json', 'summary'),
                               (f'pruning_mips/{cid}_pruning_reasons.json', 'nodes')]:
            path = (root / relative).resolve()
            if root not in path.parents:
                errors.append('原因记录路径超出实验目录')
                continue
            if not path.is_file():
                continue
            try:
                report = json.loads(path.read_text())
                if kind == 'summary':
                    values = report['removal_counts']
                    if not isinstance(values, dict) or any(
                        not isinstance(k, str) or not k or type(v) is not int or v < 0
                        for k, v in values.items()
                    ):
                        raise ValueError('原因计数格式异常')
                    counts = {k: v for k, v in values.items() if v > 0}
                else:
                    nodes = report['removed_nodes']
                    if not isinstance(nodes, dict):
                        raise ValueError('逐节点原因格式异常')
                    values = Counter()
                    for info in nodes.values():
                        code = info['code']
                        if not isinstance(code, str) or not code:
                            raise ValueError('原因代码无效')
                        values[code] += 1
                        names[code] = str(info.get('reason', code))
                    counts = dict(values)
                source = kind
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{kind}: {exc}')
        status = 'recorded' if counts is not None else 'invalid' if errors else 'missing'
        cells.append(dict(id=cid, status=status, counts=counts, source=source))
        for code, count in (counts or {}).items():
            entry = rules.setdefault(code, dict(code=code, reason=names.get(code, code), cells=0, nodes=0))
            entry['cells'] += 1
            entry['nodes'] += count
    return dict(cells=cells, rules=sorted(rules.values(), key=lambda r: (-r['cells'], r['code'])),
                total=len(cells), recorded=sum(c['status'] == 'recorded' for c in cells),
                missing=sum(c['status'] == 'missing' for c in cells),
                invalid=sum(c['status'] == 'invalid' for c in cells))
