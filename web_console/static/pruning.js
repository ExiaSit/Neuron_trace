'use strict';
/* Uses only job-scoped artifacts; never follows paths embedded in a reason report. */
const PruningReasons=(()=>{
const labels={
 initial_short_terminal:'初始末梢三维路径长度小于 7.75 µm',
 local_connection_gap:'局部三方向投影及三维原图确认连接断口',
 final_short_terminal:'最终末梢三维路径长度小于 7.75 µm',
 soma_adjacent_dead_end:'胞体外短末端缺少独立管状信号，或重复贴合保留分支',
 qc_merge_close_point:'删除距离过近的冗余节点',qc_remove_zigzag:'连续两次急转形成 Zigzag',
 qc_remove_bad_binary_bifurcation:'异常二叉分叉',qc_remove_branch_towards_soma:'分支重新朝向胞体',
 sustained_3d_signal_gap:'连续三维路径缺少横截面信号支持',switch_to_transverse_shaft:'急转接入另一条三维分支',
 sustained_sheet_like_signal:'连续三维路径沿宽片状亮信号延伸，缺少横向局限',
 remove_sharp_bends:'二级及以上分支局部急转',remove_crosstalk:'远端分支曲折度过大',
 remove_branches_entering_neighbor_soma:'分支进入其他 neuron 的胞体',
 check_root_ids_connectivity_in_region:'分支根节点与胞体连通域不连通',
 'get_soma_mask_inside_ids / soma sphere cut':'节点位于胞体掩膜内部',
 'process_neuron_morphology: soma cut and reconnection':'胞体裁剪与重连后未保留',
 'process_neuron_morphology: dark-valley filter':'分支中间出现深暗谷值',
 'process_neuron_morphology: soma-boundary direction check':'胞体出口方向朝向内部',
 'NeuronGraphProcessor._check_angle_and_radius':'分支半径绝对值过大',
 'NeuronGraphProcessor._check_intensity_with_parent':'相邻长段亮度突然升高',
 'NeuronGraphProcessor._detect_rebound':'长程亮度异常反弹',
 'NeuronGraphProcessor._check_intensity_and_radius':'当前亮度高于前两代分支'
};
function parseSWC(text){
 const nodes=new Map();
 for(const line of text.split(/\r?\n/)){
  if(!line.trim()||line.trim().startsWith('#'))continue;
  const n=line.trim().split(/\s+/).slice(0,7).map(Number);
  if(n.length!==7||!n.every(Number.isFinite)||!Number.isInteger(n[0])||nodes.has(n[0]))throw Error('SWC 节点格式异常');
  nodes.set(n[0],n);
 }
 if(!nodes.size)throw Error('SWC 没有有效节点');
 return nodes;
}
function model(report,input,output){
 const groups=new Map(),removed=new Map();let missingRecords=0,unmatched=0;
 for(const n of input.values()){
  if(output.has(n[0]))continue;
  const info=report.removed_nodes?.[String(n[0])];
  if(!info)missingRecords++;
  const code=info?.code||'unrecorded',reason=info?.reason||'未记录具体原因';
  const key=JSON.stringify([code,reason]);
  if(!groups.has(key))groups.set(key,{key,code,reason,count:0,length:0,roots:0});
  const g=groups.get(key);g.count++;
  const parent=input.get(n[6]);
  if(parent)g.length+=Math.hypot(n[2]-parent[2],n[3]-parent[3],n[4]-parent[4]);
  removed.set(n[0],key);
 }
 for(const [id,key] of removed){if(removed.get(input.get(id)[6])!==key)groups.get(key).roots++;}
 for(const id of Object.keys(report.removed_nodes||{})){if(!input.has(Number(id))||output.has(Number(id)))unmatched++;}
 return {input,output,removed,groups:[...groups.values()].sort((a,b)=>b.count-a.count),missingRecords,unmatched,summaryOnly:false};
}
function summaryModel(report){
 return {summaryOnly:true,groups:Object.entries(report.removal_counts||{}).map(([code,count])=>({key:code,code,reason:labels[code]||code,count,length:null,roots:null})).sort((a,b)=>b.count-a.count)};
}
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
const svgEl=(tag,attrs={})=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;};
class View{
 constructor(){
  this.box=document.querySelector('#pruning-filter');this.select=document.querySelector('#pruning-reason');this.key='';this.serial=0;
  this.select.onchange=()=>{this.wantedCode=null;this.render();};
  document.querySelector('#pruning-reset').onclick=()=>{this.wantedCode=null;this.select.value='all';this.render();};
 }
 focusReason(code){this.wantedCode=code;const group=this.data?.groups.find(g=>g.code===code);this.select.value=group?group.key:'all';this.render();}
 visibility(){this.box.hidden=!this.available||!document.querySelector('#result-show-pruning_mips').checked;}
 async load(items,url){
  const local=items.filter(a=>!a.scope),report=local.find(a=>a.name.endsWith('_pruning_reasons.json')),
   summary=local.find(a=>a.name.endsWith('_pruning-audit.json')),
   input=local.find(a=>a.group==='gcut_output/selected_resampled_step1'&&a.path.endsWith('.swc')),
   output=local.find(a=>a.group==='pruning_swcs'&&a.path.endsWith('.swc'));
  // Capture URLs before awaiting: the user may switch neuron or experiment mid-request.
  const urls=[report,summary,input,output].map(a=>a?url(a):null),key=JSON.stringify(urls);
  this.available=Boolean(report||summary||local.some(a=>a.group==='pruning_mips'));this.visibility();
  if(this.key===key)return;this.key=key;const serial=++this.serial;this.data=null;
  document.querySelector('#pruning-projections').replaceChildren();document.querySelector('#pruning-stats').replaceChildren();
  this.select.replaceChildren(new Option('全部原因','all'));this.select.disabled=true;
  const status=document.querySelector('#pruning-status');status.textContent='正在读取剪枝原因…';
  const get=async(u,json)=>{const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw Error(`读取失败（${r.status}）`);return json?r.json():r.text();};
  try{
   let data;
   if(report&&input&&output){const [r,a,b]=await Promise.all([get(urls[0],true),get(urls[2],false),get(urls[3],false)]);data=model(r,parseSWC(a),parseSWC(b));}
   else if(summary)data=summaryModel(await get(urls[1],true));
   else{if(serial===this.serial)status.textContent='没有完整的剪枝原因与前后 SWC 记录，暂时无法筛选；阶段完成后可刷新。';return;}
   if(serial!==this.serial)return;
   this.data=data;this.select.replaceChildren(new Option('全部原因','all'),...data.groups.map(g=>new Option(`${g.reason}（${g.count} 节点）`,g.key)));
   this.select.disabled=!data.groups.length;this.focusReason(this.wantedCode);
  }catch(e){if(serial===this.serial){status.textContent='剪枝原因加载失败：'+e.message+'。请点击“刷新结果”重试。';this.key='';}}
 }
 render(){
  const d=this.data;if(!d)return;
  const selected=this.select.value,groups=d.groups.filter(g=>selected==='all'||g.key===selected),count=groups.reduce((s,g)=>s+g.count,0);
  document.querySelector('#pruning-status').textContent=d.summaryOnly?
   `选中 ${count.toLocaleString()} 个删除节点。此历史结果仅保留原因汇总，无法恢复各原因的空间位置和路径长度。新实验会保留逐节点记录。`:
   `选中 ${count.toLocaleString()} / ${d.removed.size.toLocaleString()} 个删除节点。红色：选中原因；灰色：剪枝前结构；青色：剪枝后结构；黄色：同原因删除片段的起点。三方向为 SWC 投影，未叠加原图。`+
   (d.missingRecords?` ${d.missingRecords} 个节点未记录具体原因，单独归类。`:'')+(d.unmatched?` ${d.unmatched} 条原因记录无法对应本次前后 SWC，未计入。`:'');
  const stats=document.querySelector('#pruning-stats');stats.replaceChildren();
  const table=el('table'),head=el('tr');for(const x of ['原因','删除节点数','原始边累计长度（µm）'])head.append(el('th',x));table.append(head);
  for(const g of groups){const tr=el('tr'),td=el('td'),details=el('details');details.append(el('summary',g.reason),el('code',g.code));td.append(details);tr.append(td,el('td',String(g.count)),el('td',g.length===null?'未保留':g.length.toFixed(2)));table.append(tr);}
  stats.append(table,el('p','长度为被删节点在剪枝输入 SWC 中的父边之和，包含连接到保留节点的边；不等于最终总长度之差，也不等于整条分支数。'));
  const plots=document.querySelector('#pruning-projections');plots.replaceChildren();if(d.summaryOnly)return;
  for(const [name,x,y] of [['XY',2,3],['XZ',2,4],['YZ',3,4]]){
   const figure=el('figure'),caption=el('figcaption',name+' · µm'),svg=svgEl('svg',{role:'img','aria-label':name+' 剪枝原因投影',preserveAspectRatio:'xMidYMid meet'});
   let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
   for(const nodes of [d.input,d.output])for(const n of nodes.values()){xmin=Math.min(xmin,n[x]);xmax=Math.max(xmax,n[x]);ymin=Math.min(ymin,n[y]);ymax=Math.max(ymax,n[y]);}
   const pad=Math.max(xmax-xmin,ymax-ymin,1)*.04;svg.setAttribute('viewBox',`${xmin-pad} ${ymin-pad} ${xmax-xmin+2*pad} ${ymax-ymin+2*pad}`);
   const path=(nodes,color,width,predicate)=>{let commands='';for(const n of nodes.values()){const p=nodes.get(n[6]);if(p&&(!predicate||predicate(n)))commands+=`M${p[x]},${p[y]}L${n[x]},${n[y]}`;}svg.append(svgEl('path',{d:commands,fill:'none',stroke:color,'stroke-width':width,'vector-effect':'non-scaling-stroke'}));};
   path(d.input,'#566369',1);path(d.output,'#38d9ff',1.3);path(d.input,'#ff7164',2,n=>d.removed.has(n[0])&&(selected==='all'||d.removed.get(n[0])===selected));
   for(const [id,key] of d.removed){if(selected!=='all'&&selected!==key)continue;const n=d.input.get(id);if(d.removed.get(n[6])===key)continue;const c=svgEl('circle',{cx:n[x],cy:n[y],r:pad*.13,fill:'#ffdc70'}),title=svgEl('title');title.textContent=`节点 ${id} · ${d.groups.find(g=>g.key===key).reason}`;c.append(title);svg.append(c);}
   figure.append(caption,svg);plots.append(figure);
  }
 }
}
function matchingCells(data,code,minimum=1){
 return data.cells.filter(c=>c.status==='recorded'&&(c.counts?.[code]||0)>=minimum)
  .sort((a,b)=>b.counts[code]-a.counts[code]||Number(a.id)-Number(b.id));
}
class CohortView{
 constructor(api,onSelect){
  this.api=api;this.onSelect=onSelect;this.serial=0;
  this.select=document.querySelector('#neuron-rule');this.minimum=document.querySelector('#neuron-rule-min');
  this.status=document.querySelector('#neuron-rule-status');this.matches=document.querySelector('#neuron-rule-matches');
  this.select.onchange=()=>this.render();this.minimum.oninput=()=>this.render();
  document.querySelector('#neuron-rule-refresh').onclick=()=>{if(this.job)this.load(this.job);};
  document.querySelector('#neuron-rule-clear').onclick=()=>{this.select.value='';this.minimum.value='1';this.render();};
  this.reset();
 }
 reset(){this.serial++;this.job=null;this.data=null;this.select.replaceChildren(new Option('请选择剪枝规则',''));this.select.disabled=true;this.minimum.value='1';this.matches.replaceChildren();this.status.textContent='';}
 async load(id){
  const previous=this.job===id?this.select.value:'';
  this.job=id;const serial=++this.serial;this.select.disabled=true;this.data=null;this.matches.replaceChildren();this.status.textContent='正在汇总当前实验的剪枝规则…';
  try{
   const data=await this.api('/api/pruning-index?id='+encodeURIComponent(id));if(serial!==this.serial)return;
   this.data=data;this.select.replaceChildren(new Option('请选择剪枝规则',''),...data.rules.map(r=>new Option(`${labels[r.code]||r.reason}（${r.cells} 例）`,r.code)));
   this.select.value=data.rules.some(r=>r.code===previous)?previous:'';this.select.disabled=!data.rules.length;this.render();
  }catch(e){if(serial===this.serial)this.status.textContent='筛选数据加载失败：'+e.message+'。可点击刷新重试。';}
 }
 render(){
  const data=this.data;if(!data)return;this.matches.replaceChildren();
  const coverage=`共 ${data.total} 例，${data.recorded} 例有记录，${data.missing} 例无记录，${data.invalid} 例记录异常。`;
  const code=this.select.value;
  if(!code){this.status.textContent=coverage+(data.rules.length?'选择规则后显示命中的 neuron。':'现有记录中没有触发剪枝规则的样本。');return;}
  const minimum=Number(this.minimum.value);
  if(!Number.isInteger(minimum)||minimum<1){this.status.textContent='最少删除节点数请输入大于等于 1 的整数。';return;}
  const cells=matchingCells(data,code,minimum);
  this.status.textContent=`匹配 ${cells.length} 例（该规则至少删除 ${minimum} 个节点）。`+coverage+' 点击编号查看该样本；按删除节点数从多到少排列。';
  for(const cell of cells){const button=el('button',`${cell.id} · ${cell.counts[code]} 节点`);button.type='button';button.onclick=()=>this.onSelect(cell.id,code);this.matches.append(button);}
 }
}
return {parseSWC,model,summaryModel,View,matchingCells,CohortView};
})();
if(typeof module!=='undefined')module.exports=PruningReasons;
