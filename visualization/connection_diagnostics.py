"""Pre-cut evidence sheets: identical 3-D crop for XY, XZ and YZ projections."""
from pathlib import Path
import html
import numpy as np
from collections import deque

_executor=None
_pending=deque()
_queue_limit=1


def _initialize_renderer():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def configure_render_workers(workers=1):
    """Parallelize only figure export; diagnostic decisions remain sequential."""
    global _executor,_queue_limit
    finish_rendering()
    if workers>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        _executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                      initializer=_initialize_renderer)
        _queue_limit=workers*2


def finish_rendering():
    global _executor
    try:
        while _pending:_pending.popleft().result()
    finally:
        if _executor is not None:_executor.shutdown(wait=True,cancel_futures=True)
        _executor=None


def render_site(crop, lo_xyz, site, output):
    if _executor is None:return _render_site_sync(crop,lo_xyz,site,output)
    if len(_pending)>=_queue_limit:_pending.popleft().result()
    _pending.append(_executor.submit(_render_site_sync,crop,lo_xyz,site,output))


def _render_site_sync(crop, lo_xyz, site, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'supported':'#3fe39a','disconnected':'#ff6262','uncertain':'#ffd34d','suspected_switch':'#ff80df'}
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    lo=np.asarray(lo_xyz); hi=lo+np.array(crop.shape)[::-1]
    for col,(plane,axis,xy) in enumerate([('XY',0,(0,1)),('XZ',1,(0,2)),('YZ',2,(1,2))]):
        mip=crop.max(axis=axis)
        extent=(lo[xy[0]]-.5,hi[xy[0]]-.5,hi[xy[1]]-.5,lo[xy[1]]-.5)
        vmax=max(15,float(np.percentile(mip,99.5)))
        for row in range(2):
            ax=axes[row,col];ax.imshow(mip,cmap='gray',vmin=0,vmax=vmax,extent=extent,interpolation='nearest')
            ax.set_xlabel('XYZ'[xy[0]]+' (um)');ax.set_ylabel('XYZ'[xy[1]]+' (um)')
            ax.set_title(plane+(' raw MIP' if row==0 else ' SWC before cuts'))
        ax=axes[1,col]
        context=np.asarray(site['context_xyz'])
        if len(context):ax.plot(context[:,xy[0]],context[:,xy[1]],color='#48dfff',lw=1.5,label='Incoming')
        for key,color,label in [('probe_opposite_xyz','#80aaff','Opposite shaft probe'),
                                ('probe_ahead_xyz','#ffae60','Incoming extension probe')]:
            if key in site:
                probe=np.asarray(site[key]);ax.plot(probe[:,xy[0]],probe[:,xy[1]],'--',color=color,lw=1.5,label=label)
        for index,arm in enumerate(site['arms'],1):
            p=np.asarray(arm['xyz']);d=arm['diagnosis'];color=colors[d['status']]
            ax.plot(p[:,xy[0]],p[:,xy[1]],color=color,lw=1.5,label=f"Arm {index}: {d['status']}")
            ax.text(p[-1,xy[0]],p[-1,xy[1]],str(index),color=color,fontsize=11,
                    bbox=dict(facecolor='black',alpha=.6,pad=1))
            if d.get('cut_xyz'):
                c=d['cut_xyz'];ax.scatter(c[xy[0]],c[xy[1]],s=65,marker='x',color='#ff6262',zorder=10)
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
    axes[1,0].legend(loc='best',fontsize=7,facecolor='#222222',labelcolor='white')
    bbox=' / '.join(f'{axis} [{a}, {b})' for axis,(a,b) in zip('XYZ',site['bbox_xyz']))
    subtitle=site.get('connection_interpretation','')
    fig.suptitle(f"Site {site['site_id']} | {site['kind']} | {site['status']}\n{bbox}\n{subtitle}",fontsize=12)
    fig.savefig(output,dpi=130);plt.close(fig)
    from visualization.projection_viewer import write_site_viewer
    write_site_viewer(crop, lo_xyz, site, Path(output).with_suffix('.html'))


def write_gallery(report, output_dir):
    # diagnose_then_cut cannot proceed until every PNG/HTML is on disk.
    finish_rendering()
    items=[]
    for s in report['sites']:
        detail=[]
        for i,a in enumerate(s['arms'],1):
            d=a.get('diagnosis',{})
            detail.append(f"分支 {i}：{d.get('status','uncertain')}；{d.get('reason','outside_image')}")
        items.append(f'<section><h2>{s["site_id"]} · {html.escape(s["kind"])} · {html.escape(s["status"])}</h2>'
                     f'<p>拟剪节点：{s.get("proposed_cut_ids",[])}</p><p>'+html.escape('；'.join(detail))+'</p>'
                     +(f'<a href="{s["image"]}"><img loading="lazy" src="{s["image"]}" alt="三方向连接诊断"></a>' if s.get('image') else '')+'</section>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部连接诊断</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 24px;background:#f4f5f7;color:#17212d}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}h2{font-size:19px}</style>
<h1>局部包围盒与三方向投影诊断（剪断前）</h1>
<p>每张图的上排为原图投影，下排叠加剪断前的局部分支。绿色：影像支持；黄色：证据不足；红色：检出断口；粉色：疑似转入其他纤维。诊断图和 JSON 先保存，然后执行剪断。</p>
<p>断口需局部两侧亮信号、三维连续弱段 ≥3 µm 且缺少局部线状信号，并有至少两个随路径移动的 3 µm 薄层投影支持。完整长分支使用重叠窗口补查。单纯变暗、转弯或投影交错不删除。</p>
<p>串线规则单独记录为归属推断：非分叉处转向 ≥60°，来路、去路及去路反向延伸均有三维线状信号，原方向前方缺少信号，且至少两个有效薄层投影支持。蓝色虚线为反向纤维探针，橙色虚线为原方向前方探针。这不是生物学归属真值，需优先复核粉色位置。</p>
<p><a href="interactive.html">薄层滑动 MIP / 逐层切片 / 可调窗口</a> · <a href="diagnosis.json">全部诊断 JSON</a> · <a href="cuts.json">实际剪断记录</a></p>'''
    document+=f'<p>位置数：{len(report["sites"])}；诊断统计：{html.escape(str(report["counts"]))}</p>'
    (Path(output_dir)/'index.html').write_text(document+''.join(items)+'</html>',encoding='utf-8')
    from visualization.projection_viewer import write_viewer_index
    write_viewer_index(report, output_dir)
