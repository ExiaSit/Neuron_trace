"""Original intensity evidence with shared world XYZ soma point labels."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PLANES=[('XY',0,(0,1)),('XZ',1,(0,2)),('YZ',2,(1,2))]

def bounds(raw,points,pad):
    p=np.array(points);return np.maximum(0,np.floor(p.min(0)-pad)).astype(int),np.minimum(np.array(raw.shape[::-1])-1,np.ceil(p.max(0)+pad)).astype(int)

def render(raw,candidate,references,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cid=candidate['candidate_id'];center=np.array(candidate['xyz_um']);marks={cid:center.tolist(),**{r['candidate_id']:r['xyz_um'] for r in references}}
    views={};images=[]
    def panel(ax,lo,hi,plane,axis,xy,name,window):
        crop=raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1];a,b=xy;ax.imshow(crop.max(axis),cmap='gray',vmin=window[0],vmax=window[1],interpolation='nearest',extent=[lo[a]-.5,hi[a]+.5,hi[b]+.5,lo[b]-.5]);visible={}
        for label,p in marks.items():
            if not (lo[a]-.5<=p[a]<=hi[a]+.5 and lo[b]-.5<=p[b]<=hi[b]+.5):continue
            in_depth=lo[2-axis]-.5<=p[2-axis]<=hi[2-axis]+.5
            if not in_depth and 'slice' not in name:continue
            color='#55eaff' if label==cid else '#ffb657';ax.plot(p[a],p[b],'+',color=color,ms=5)
            ax.annotate(label,(p[a],p[b]),xytext=(4,5 if label==cid else -9),textcoords='offset points',fontsize=7,color=color)
            visible[label]=dict(xyz_um=p,in_depth=bool(in_depth),uv_um=[p[a],p[b]])
        ax.set(xlim=(lo[a]-.5,hi[a]+.5),ylim=(hi[b]+.5,lo[b]-.5),xlabel='XYZ'[a]+' (um)',ylabel='XYZ'[b]+' (um)',aspect='equal');ax.set_title(name,fontsize=8);ax.tick_params(labelsize=7)
        return dict(view_id=name,raw=True,plane=plane,xyz_bounds=[lo.tolist(),hi.tolist()],crop_origin_xyz=lo.tolist(),array_projection_axis=axis,landmarks=visible,window=window,world_axes=list(xy))
    def finish(fig,path,records):
        fig.canvas.draw();fig.set_layout_engine(None);width,height=fig.canvas.get_width_height()
        for name,v,ax,xy in records:
            origin=ax.transData.transform([0,0]);u=ax.transData.transform([1,0])-origin;w=ax.transData.transform([0,1])-origin;matrix=np.zeros((2,4));matrix[0,xy[0]]=u[0];matrix[0,xy[1]]=w[0];matrix[0,3]=origin[0];matrix[1,xy[0]]=-u[1];matrix[1,xy[1]]=-w[1];matrix[1,3]=height-origin[1]
            v.update(image=str(path),xyz_to_png_uv=matrix.tolist(),image_size_wh=[width,height]);views[name]=v
        fig.savefig(path);plt.close(fig);images.append(str(path))
    def window(lo,hi):
        c=raw[lo[2]:hi[2]+1,lo[1]:hi[1]+1,lo[0]:hi[0]+1];low=float(np.percentile(c,2));return [low,max(low+1,min(255,max(30,float(np.percentile(c,99.5)))))]
    local_lo,local_hi=bounds(raw,[center],22)
    for scope,points,pad in [('local',[center],22),('comparison',list(marks.values()),12)]:
        lo,hi=bounds(raw,points,pad);w=window(lo,hi);fig,axes=plt.subplots(1,3,figsize=(12,4.4),dpi=180,layout='constrained');records=[]
        for ax,(plane,axis,xy) in zip(axes,PLANES):
            name=scope+'_'+plane;v=panel(ax,lo,hi,plane,axis,xy,name,w);records.append((name,v,ax,xy))
        fig.suptitle(f'{cid}: RAW intensity | cyan=current; orange=retained reference | identical IDs = identical 3D points',fontsize=9);finish(fig,out/(scope+'.png'),records)
    fig,axes=plt.subplots(3,5,figsize=(15,9),dpi=180,layout='constrained');records=[];w=window(local_lo,local_hi)
    for row,(plane,axis,xy) in enumerate(PLANES):
        d=2-axis
        for col,offset in enumerate([-2,-1,0,1,2]):
            depth=int(np.clip(round(center[d])+offset,local_lo[d],local_hi[d]));lo=local_lo.copy();hi=local_hi.copy();lo[d]=hi[d]=depth;name=f'slice_{plane}_{depth}'
            v=panel(axes[row,col],lo,hi,plane,axis,xy,name,w);records.append((name,v,axes[row,col],xy))
    fig.suptitle('RAW five consecutive 1 um sections, center -2 through +2 um; crosses project reference coordinates onto each section',fontsize=9);finish(fig,out/'slices.png',records)
    return images,views

def overview(raw,bundle,outcomes,path):
    fig,axes=plt.subplots(1,3,figsize=(16,6),dpi=160,layout='constrained');projection=raw.max(0)
    nms=[(f"old >{bundle['baseline_threshold']:g} + 6 um NMS",bundle['baseline_nms_zyx']),(f"relaxed >{bundle['relaxed_threshold']:g} + 6 um NMS (control)",bundle['relaxed_nms_zyx'])]
    for ax,(name,points) in zip(axes,nms):
        ax.imshow(projection,cmap='gray',vmin=0,vmax=180);ax.set_title(name)
        for p in points:ax.plot(p[2],p[1],'o',mfc='none',mec='#00ffff',ms=7)
    ax=axes[2];ax.imshow(projection,cmap='gray',vmin=0,vmax=180);ax.set_title('Gemma proposals\ngreen: accept | orange: duplicate\nred: reject | yellow: review',fontsize=11)
    for c in bundle['candidates']:
        d=outcomes.get(c['candidate_id'],{}).get('decision',{}).get('decision','REVIEW');color={'ACCEPT':'#00ff77','DUPLICATE':'#ff9f44','REJECT':'#ff5555','REVIEW':'#ffff33'}[d];x,y,z=c['xyz_um'];ax.plot(x,y,'o',mfc='none',mec=color,ms=5);ax.annotate(c['candidate_id'],(x,y),fontsize=5,color=color)
    for ax in axes:
        t=bundle['target']['xyz_um'];ax.plot(t[0],t[1],'+',color='cyan',ms=10);ax.set(xlabel='X (um)',ylabel='Y (um)')
    fig.suptitle('Soma seed detection only; no G-Cut split, no pruning; outputs are not ground truth');fig.savefig(path);plt.close(fig)
