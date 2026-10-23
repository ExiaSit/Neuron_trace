'use strict';
// Shares the existing authenticated API session; no local uploads or file reads.
const picker={input:null,kind:'directory',path:'',parent:null,selected:null,offset:0,limit:100,total:0,sequence:0,timer:null};
const pathLabels={previous_app2_dir:'旧版 APP2 目录',metadata_file:'元数据 CSV',v3d_root:'V3D 根目录',v3d_dir:'具体 V3D 文件目录',source_tif_dir:'原始分辨率 TIFF 目录',raw_dir:'1 µm 灰度原图目录',mask_dir:'初始 mask 目录',soma_crop_dir:'胞体裁剪 / 位置 JSON 目录',soma_seg_dir:'胞体分割目录',neurite_seg_dir:'神经突分割目录',upstream_root:'已有重建结果根目录'};

function pickerSelection(path){
  picker.selected=path;
  $('#picker-selection').textContent=path||(picker.kind==='file'?'尚未选择 CSV 文件':'尚未选择文件夹');
  $('#picker-confirm').disabled=!path;
  for(const entry of $('#picker-list').querySelectorAll('[data-file]')){
    const chosen=entry.dataset.file===path;
    entry.classList.toggle('chosen',chosen);entry.setAttribute('aria-pressed',String(chosen));
  }
}
function pickerCrumbs(data){
  const nav=$('#picker-breadcrumbs');nav.replaceChildren();
  const root=data.roots.filter(r=>data.path===r||data.path.startsWith(r+'/')).sort((a,b)=>b.length-a.length)[0];
  if(!root)return;
  let path=root;
  const add=(label,target)=>{const button=node('button',label);button.type='button';button.onclick=()=>browsePath(target);nav.append(button);};
  add(root,root);
  for(const part of data.path.slice(root.length).split('/').filter(Boolean)){
    path+='/'+part;nav.append(node('span','/'));add(part,path);
  }
}
async function browsePath(path,offset=0,query=''){
  clearTimeout(picker.timer);
  const sequence=++picker.sequence;
  pickerSelection(null);$('#picker-error').hidden=true;
  $('#picker-list').setAttribute('aria-busy','true');$('#picker-count').textContent='正在读取服务器目录…';
  $('#picker-prev').disabled=true;$('#picker-next').disabled=true;
  try{
    const q=new URLSearchParams({path,kind:picker.kind,q:query,offset:String(offset)});
    const data=await api('/api/browse?'+q);
    if(sequence!==picker.sequence||!$('#path-picker').open)return;
    Object.assign(picker,{path:data.path,parent:data.parent,offset:data.offset,limit:data.limit,total:data.total});
    $('#picker-location').value=data.path;$('#picker-search').value=query;$('#picker-up').disabled=!data.parent;
    $('#picker-roots').replaceChildren(...data.roots.map(root=>{const b=node('button',root);b.type='button';b.onclick=()=>browsePath(root);return b;}));
    pickerCrumbs(data);
    const list=$('#picker-list');list.replaceChildren();
    for(const entry of data.entries){
      const button=node('button',undefined,'picker-entry');button.type='button';
      button.append(node('span',entry.directory?'目录':'CSV','entry-icon'),node('span',entry.name+(entry.symlink?' ↗':''),'entry-name'),node('span',entry.directory?'进入 →':'选择','entry-action'));
      if(entry.directory){button.onclick=()=>browsePath(entry.path);button.setAttribute('aria-label','进入目录 '+entry.name);}
      else{button.dataset.file=entry.path;button.setAttribute('aria-pressed','false');button.onclick=()=>pickerSelection(entry.path);}
      list.append(button);
    }
    if(!data.entries.length){list.append(node('div',query?'没有匹配的名称，试试其他关键词。':picker.kind==='file'?'此目录没有 CSV 文件或可进入的子目录。':'此目录没有可进入的子目录，可以直接选择当前文件夹。','empty'));}
    pickerSelection(picker.kind==='directory'?data.path:data.selected);
    $('#picker-count').textContent=data.total?`共 ${data.total} 项 · ${data.offset+1}–${Math.min(data.offset+data.limit,data.total)}`:'0 项';
    $('#picker-prev').disabled=data.offset===0;$('#picker-next').disabled=!data.has_more;
    list.scrollTop=0;
  }catch(error){
    if(sequence!==picker.sequence||!$('#path-picker').open)return;
    $('#picker-error').textContent=error.message;$('#picker-error').hidden=false;
    $('#picker-list').replaceChildren(node('div','可从上方常用目录重新进入，或修改地址后点击“前往”。','empty'));
    $('#picker-count').textContent='目录读取失败';
  }finally{if(sequence===picker.sequence)$('#picker-list').setAttribute('aria-busy','false');}
}
function openPicker(input){
  if(!connected){notice('请先连接工作台，再浏览服务器目录。',true);$('#token').focus();return;}
  picker.input=input;picker.kind=input.name==='metadata_file'?'file':'directory';
  $('#picker-title').textContent='选择'+pathLabels[input.name];
  $('#picker-help').textContent=picker.kind==='file'?'浏览服务器目录，点击 CSV 文件后确认选择。':'点击目录进入；到达目标目录后，点击“选择此文件夹”。';
  $('#picker-confirm').textContent=picker.kind==='file'?'选择此文件':'选择此文件夹';
  $('#picker-selection-label').textContent=picker.kind==='file'?'已选文件':'当前文件夹';
  $('#picker-list').replaceChildren();$('#picker-breadcrumbs').replaceChildren();
  $('#picker-search').value='';$('#picker-location').value=input.value.trim();picker.parent=null;$('#picker-up').disabled=true;
  // Show root shortcuts even when a pasted path is invalid.
  $('#picker-roots').replaceChildren(...['/data','/mnt','/media','/home/pzy'].map(root=>{const b=node('button',root);b.type='button';b.onclick=()=>browsePath(root);return b;}));
  $('#path-picker').showModal();browsePath(input.value.trim());
}
for(const input of document.querySelectorAll('.paths input')){
  const row=node('div',undefined,'path-input-row');
  const button=node('button','浏览服务器','path-browse');button.type='button';
  button.setAttribute('aria-label','浏览服务器并选择'+pathLabels[input.name]);
  input.id=input.id||'input-'+input.name;input.closest('label').htmlFor=input.id;
  input.before(row);row.append(input,button);button.onclick=()=>openPicker(input);
}
$('#picker-close').onclick=$('#picker-cancel').onclick=()=>$('#path-picker').close();
$('#path-picker').addEventListener('close',()=>{++picker.sequence;clearTimeout(picker.timer);picker.input?.focus();});
$('#picker-up').onclick=()=>{if(picker.parent)browsePath(picker.parent);};
$('#picker-location-form').onsubmit=e=>{e.preventDefault();browsePath($('#picker-location').value.trim());};
$('#picker-search').oninput=()=>{
  clearTimeout(picker.timer);
  // Invalidate an older request immediately so it cannot erase new search text.
  ++picker.sequence;pickerSelection(null);
  picker.timer=setTimeout(()=>browsePath(picker.path,0,$('#picker-search').value.trim()),250);
};
$('#picker-prev').onclick=()=>browsePath(picker.path,Math.max(0,picker.offset-picker.limit),$('#picker-search').value.trim());
$('#picker-next').onclick=()=>browsePath(picker.path,picker.offset+picker.limit,$('#picker-search').value.trim());
$('#picker-confirm').onclick=()=>{
  if(!picker.selected||!picker.input)return;
  picker.input.value=picker.selected;
  picker.input.dispatchEvent(new Event('input',{bubbles:true}));
  picker.input.dispatchEvent(new Event('change',{bubbles:true}));
  $('#path-picker').close();
};
