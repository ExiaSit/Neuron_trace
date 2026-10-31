/* Result download groups. Keep download names and authenticated URLs unchanged. */
(function(root,factory){
 const api=factory();
 if(typeof module==='object'&&module.exports)module.exports=api;
 else root.NeuronResources=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 const groups=[
  ['preparation','01–06 · 前处理'],['mask','07 · Mask 修复'],
  ['app2','08 · APP2 追踪'],['gcut','09 · G-Cut 分离'],
  ['resample','10 · SWC 重采样'],['pruning','11 · 后续剪枝'],
  ['guard','12 · 回归保护'],['comparison','13 · 最终比较'],
  ['paired_case','版本对照 · 当前 neuron'],['paired_batch','版本对照 · 整批统计'],
  ['other','其他结果']
 ];
 const maskNames=['骨架及方向分析','三维信号验证与初次重连','胞体补全','胞体与分支重连','连接厚度恢复','胞体根部补全','受约束延伸','根部测地线补全'];
 function describe(a){
  const p=a.path||'',g=a.group;
  if(g==='mip_comparison'){
   if(p.startsWith('baseline/pruning_swcs/'))return {group:'paired_case',role:'v0.83 最终 SWC'};
   if(p.startsWith('candidate/pruning_swcs/'))return {group:'paired_case',role:'v0.83a 最终 SWC'};
   if(p.startsWith('cases/'))return {group:'paired_case',role:'本例对照指标'};
   if(p.startsWith('branch_audits/'))return {group:'paired_case',role:'本例分支保留核查'};
   if(p.startsWith('comparisons/'))return {group:'paired_case',role:'三方向 Mask 对照'};
   return {group:'paired_batch',role:({'per_case.csv':'逐例指标','summary.json':'队列汇总','aggregate_metrics.json':'重建指标汇总','branch_retention.json':'分支保留统计','summary_metrics.png':'总体统计图'})[p]||'整批统计'};
  }
  if(g==='mask_step_mips'){
   const match=p.match(/_(\d{2})_([a-h])\./);
   return {group:'mask',role:match?`${match[2]} · ${maskNames[Number(match[1])-1]||'修复记录'}`:'Mask 步骤记录'};
  }
  if(g==='mask_original')return {group:'mask',role:'原始 Mask（补全前，3D TIFF）'};
  if(g==='mask')return {group:'mask',role:'补全后 Mask（3D TIFF）'};
  if(g==='mask_mips')return {group:'mask',role:'Mask 修复前后'};
  if(g==='preparation_mips')return {group:'preparation',role:'前处理记录'};
  if(g==='trace_app2/down_sampled_swcs_app2')return {group:'app2',role:'APP2 追踪 SWC'};
  if(g==='app2_mips')return {group:'app2',role:'APP2 追踪图'};
  if(g==='app2_input')return {group:'app2',role:'骨架化实验输入（3D TIFF）'};
  if(g==='gcut_output/selected_for_pruning')return {group:'gcut',role:'G-Cut 目标 SWC'};
  if(g==='gcut_audits')return {group:'gcut',role:p.endsWith('_before.swc')?'短末梢删除前 SWC':p.endsWith('_after.swc')?'短末梢删除后 SWC':'短末梢删除记录'};
  if(g==='gcut_mips')return {group:'gcut',role:'G-Cut 分离记录'};
  if(g==='gcut_output/selected_resampled_step1')return {group:'resample',role:'1 µm 重采样 SWC'};
  if(g==='pruning_swcs')return {group:'pruning',role:'最终剪枝 SWC'};
  if(g==='pruning_mips')return {group:'pruning',role:p.endsWith('pruning-audit.json')?'剪枝原因统计':p.endsWith('_pruning_reasons.json')?'逐节点剪枝原因':'剪枝过程记录'};
  if(g==='regression_guard')return {group:'guard',role:'回归保护记录'};
  if(g==='comparison_sources')return {group:'comparison',role:'参考 APP2 来源'};
  if(g==='comparisons')return {group:'comparison',role:'最终比较记录'};
  return {group:'other',role:g||'结果文件'};
 }
 function tooltip(a){
  return `${describe(a).role}\n${a.storage_path?'存储路径：'+a.storage_path:'相对路径：'+a.path}`;
 }
 function grouped(items){
  const by=new Map(groups.map(([id,title])=>[id,{id,title,items:[]}]));
  for(const a of items){if(!a.image)by.get(describe(a).group).items.push(a);}
  return [...by.values()].filter(g=>g.items.length).map(g=>({...g,items:g.items.sort((a,b)=>(a.name||a.path).localeCompare(b.name||b.path,'zh-CN',{numeric:true})||a.path.localeCompare(b.path))}));
 }
 function render(container,items,urlFor,doc){
  doc=doc||document;container.replaceChildren();
  for(const group of grouped(items)){
   const section=doc.createElement('section');section.className='resource-group';section.dataset.stage=group.id;
   const header=doc.createElement('div');header.className='resource-group-heading';
   const title=doc.createElement('h3');title.textContent=group.title;
   const count=doc.createElement('span');count.className='resource-group-count';count.textContent=`${group.items.length} 个文件`;
   header.append(title,count);section.append(header);
   const row=doc.createElement('div');row.className='file-row resource-file-row';
   for(const a of group.items){
    const link=doc.createElement('a');link.className='resource-link';link.href=urlFor(a);link.download=a.name;link.title=tooltip(a);
    const role=doc.createElement('span');role.className='resource-role';role.textContent=describe(a).role;
    const name=doc.createElement('span');name.className='resource-name';name.textContent=a.name+' ↓';
    link.append(role,name);row.append(link);
   }
   section.append(row);container.append(section);
  }
 }
 return {describe,tooltip,grouped,render};
});
