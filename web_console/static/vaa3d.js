(function(root,factory){
 const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.NeuronVaa3D=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 function buildURL(bundle,selected){
  if(!bundle?.image)throw Error('缺少原始三维图像');
  const params=new URLSearchParams({image:bundle.image}),seen=new Set([bundle.image]);
  for(const c of bundle.choices||[]){
   if(!selected.has(c.path)||seen.has(c.path)||!['swc','mask'].includes(c.role))continue;
   seen.add(c.path);params.append(c.role,c.path);
  }
  return 'vaa3d://open?'+params.toString();
 }
 class View{
  constructor(doc=document,launch=url=>{location.href=url;}){
   this.doc=doc;this.launch=launch;this.bundle=null;this.context=null;this.signature='';this.selected=new Set();
   this.button=doc.querySelector('#vaa3d-open');this.options=doc.querySelector('#vaa3d-options');
   this.status=doc.querySelector('#vaa3d-status');this.raw=doc.querySelector('#vaa3d-image');
   for(const [id,mode] of [['vaa3d-select-all','all'],['vaa3d-select-none','none'],['vaa3d-select-default','default']]){
    doc.querySelector('#'+id).onclick=()=>{
     this.selected=new Set((this.bundle?.choices||[]).filter(c=>mode==='all'||(mode==='default'&&c.default)).map(c=>c.path));
     for(const label of this.options.children){const input=label.children[0];input.checked=this.selected.has(input.dataset.path);}
     this.describe();
    };
   }
   this.button.onclick=()=>{
    if(!this.bundle?.image)return;
    this.launch(buildURL(this.bundle,this.selected));
    this.status.textContent='已请求本地 Vaa3D 打开。若浏览器询问，请允许打开应用；三维加载完成后将显示 Object Manager。';
   };
  }
  clear(){this.context=null;this.signature='';this.bundle=null;this.selected.clear();this.button.disabled=true;this.raw.textContent='正在读取当前样本…';this.options.replaceChildren();this.status.textContent='';}
  update(bundle,context){
   bundle=bundle||{choices:[],unavailable:'当前服务尚未提供 Vaa3D 文件信息，请刷新页面。'};
   const signature=JSON.stringify(bundle);
   if(this.context===context&&this.signature===signature)return;
   const same=this.context===context,previous=new Set((this.bundle?.choices||[]).map(c=>c.path));
   const chosen=same?this.selected:new Set();
   this.selected=new Set((bundle.choices||[]).filter(c=>previous.has(c.path)&&same?chosen.has(c.path):c.default).map(c=>c.path));
   this.bundle=bundle;this.context=context;this.signature=signature;
   this.button.disabled=!bundle.image;
   this.raw.textContent=bundle.image?'原图（必选）：'+bundle.image:'原图不可用';this.raw.title=bundle.image||'';
   this.options.replaceChildren();
   for(const c of bundle.choices||[]){
    const label=this.doc.createElement('label'),input=this.doc.createElement('input'),span=this.doc.createElement('span');
    input.type='checkbox';input.checked=this.selected.has(c.path);input.dataset.path=c.path;
    input.onchange=()=>{if(input.checked)this.selected.add(c.path);else this.selected.delete(c.path);this.describe();};
    const title=this.doc.createElement('strong'),file=this.doc.createElement('small');
    title.textContent=c.label;file.textContent=c.path.split('/').pop();span.append(title,file);
    label.title=c.path;label.append(input,span);this.options.append(label);
   }
   this.describe();
  }
  describe(){
   if(!this.bundle?.image){this.status.textContent=this.bundle?.unavailable||'原图不可用';return;}
   const chosen=(this.bundle.choices||[]).filter(c=>this.selected.has(c.path));
   this.status.textContent=`将打开 1 份原图、${chosen.filter(c=>c.role==='swc').length} 份 SWC、${chosen.filter(c=>c.role==='mask').length} 份 Mask。`;
  }
 }
 return {buildURL,View};
});
