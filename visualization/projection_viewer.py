"""Display-only thin-slab/slice explorer; do not feed windowed images to pruning."""
from pathlib import Path
import base64
import html
import json
import numpy as np

PLANES = [('XY', 0, (0, 1)), ('XZ', 1, (0, 2)), ('YZ', 2, (1, 2))]


def project_volume(crop, axis, center, mode='slab', thickness=5):
    if axis not in (0, 1, 2) or crop.ndim != 3 or not all(crop.shape):
        raise ValueError('Expected a nonempty ZYX volume and valid projection axis')
    if mode not in ('full', 'slab', 'slice'):
        raise ValueError('Unknown projection mode')
    if thickness not in (3, 5, 9):
        raise ValueError('Supported slab thicknesses are 3, 5 and 9 um')
    center = int(np.clip(center, 0, crop.shape[axis]-1))
    radius = 0 if mode == 'slice' else thickness//2
    lo, hi = (0, crop.shape[axis]) if mode == 'full' else (
        max(0, center-radius), min(crop.shape[axis], center+radius+1))
    slices = [slice(None)]*3; slices[axis] = slice(lo, hi)
    return crop[tuple(slices)].max(axis=axis), (lo, hi)


def window_defaults(crop):
    black = int(np.floor(np.percentile(crop, 5)))
    white = max(black+1, int(np.ceil(np.percentile(crop, 99.5))))
    return [min(254, black), min(255, white)]


def window_image(image, black, white, gamma=.85):
    if white <= black or gamma <= 0:
        raise ValueError('Invalid display window or gamma')
    values = np.clip((image.astype(float)-black)/(white-black), 0, 1)
    return np.rint(255*values**gamma).astype(np.uint8)


def write_site_viewer(crop, origin_xyz, site, output):
    if crop.dtype != np.uint8:
        raise ValueError('Viewer expects the pipeline uint8 raw image; no silent rescaling')
    data = dict(shape_zyx=list(crop.shape), origin_xyz=list(map(int, origin_xyz)),
                voxels_base64=base64.b64encode(np.ascontiguousarray(crop).tobytes()).decode('ascii'),
                window=window_defaults(crop), site=site)
    template = (Path(__file__).parent/'assets/projection_viewer.html').read_text()
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    Path(output).write_text(template.replace('__DATA_JSON__', payload), encoding='utf-8')


def render_projection_comparison(crop, origin_xyz, site, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(14, 15), layout='constrained')
    origin = np.asarray(origin_xyz); upper = origin+np.asarray(crop.shape)[::-1]
    center = np.floor(np.asarray(site['center_xyz'])-origin+.5).astype(int)
    black, white = window_defaults(crop)
    descriptions = ['Original full-box MIP', 'Full-box MIP / display window',
                    '5-um slab MIP / same window', 'Single slice / same window']
    for col, (name, axis, xy) in enumerate(PLANES):
        full, _ = project_volume(crop, axis, center[2-axis], 'full')
        extent = (origin[xy[0]]-.5, upper[xy[0]]-.5, upper[xy[1]]-.5, origin[xy[1]]-.5)
        for row, mode in enumerate(['full','full','slab','slice']):
            mip, bounds = project_volume(crop, axis, center[2-axis], mode)
            ax = axes[row,col]
            if row == 0:
                ax.imshow(mip,cmap='gray',vmin=0,vmax=max(15,float(np.percentile(full,99.5))),
                          extent=extent,interpolation='nearest')
            else:
                ax.imshow(window_image(mip,black,white),cmap='gray',vmin=0,vmax=255,
                          extent=extent,interpolation='nearest')
            low, high = bounds
            ax.set_title(f'{name}: {descriptions[row]}\n{"ZYX"[axis]} layers {origin[2-axis]+low}..{origin[2-axis]+high-1}',fontsize=10)
            ax.set_xlabel('XYZ'[xy[0]]+' (um)');ax.set_ylabel('XYZ'[xy[1]]+' (um)')
    fig.suptitle(f"Site {site['site_id']}: raw signal, no SWC overlay\nRows 2-4 share black={black}, white={white}, gamma=0.85 | 1 um native display pixels",fontsize=12)
    fig.savefig(output,dpi=130);plt.close(fig)


def write_viewer_index(report, output_dir, comparisons=()):
    sites=[s for s in report['sites'] if s.get('image')]
    options=''.join(f'<option value="site_{s["site_id"]}.html">{s["site_id"]} · {html.escape(s["kind"])} · {html.escape(s["status"])}</option>' for s in sites)
    links=' · '.join(f'<a href="display_comparison_{sid}.png">位置 {sid} 显示前后对照</a>' for sid in comparisons)
    first=f'site_{sites[0]["site_id"]}.html' if sites else 'about:blank'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>薄层投影与逐层检查</title>
<style>body{{font:16px system-ui;background:#eef2f6;color:#182737;margin:22px}}h1{{font-size:25px}}select,button{{font:inherit;padding:8px;margin-right:8px}}iframe{{width:100%;height:1100px;border:1px solid #ccd8e3;border-radius:10px;background:white}}p{{line-height:1.6}}a{{color:#176bb0}}</style>
<h1>局部薄层投影与逐层检查</h1><p>共有 {len(sites)} 个位置。默认显示 5 µm 薄层；可切换 3/9 µm、逐层切片或整盒投影。三个方向可独立滑动或播放，共用可调灰度窗口。</p>
<p>这里只改变显示；源诊断、剪枝判断和 SWC 保持不变。查看薄层时，SWC 仅显示落在该层范围内的部分。</p><p>{links}</p>
<p><button id="prev">上一处</button><select id="site" aria-label="诊断位置">{options}</select><button id="next">下一处</button><a id="open" href="{first}" target="_blank">单独打开当前视图</a> · <a href="diagnosis.json">源诊断 JSON</a></p>
<iframe id="view" title="局部三视图" src="{first}"></iframe><script>
const selector=document.getElementById('site');function show(){{document.getElementById('view').src=selector.value;document.getElementById('open').href=selector.value;}}
selector.onchange=show;document.getElementById('prev').onclick=()=>{{selector.selectedIndex=Math.max(0,selector.selectedIndex-1);show();}};document.getElementById('next').onclick=()=>{{selector.selectedIndex=Math.min(selector.options.length-1,selector.selectedIndex+1);show();}};
</script></html>'''
    (Path(output_dir)/'interactive.html').write_text(doc,encoding='utf-8')
