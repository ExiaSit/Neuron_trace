'use strict';
const assert=require('assert/strict');
const fs=require('fs');
const resources=require('../web_console/static/resources.js');
const maskPair=[{group:'mask_original',path:'mask_original/42.tif',name:'42_original_mask.tif',image:false},
 {group:'mask',path:'mask/42.tif',name:'42_repaired_mask.tif',image:false}];
assert.deepEqual(resources.grouped(maskPair).map(g=>g.id),['mask']);
assert.equal(resources.grouped(maskPair)[0].items.length,2);
assert.match(resources.describe(maskPair[0]).role,/原始 Mask/);
assert.match(resources.describe(maskPair[1]).role,/补全后 Mask/);
function item(group,path,name='60575.swc',extra={}){return {group,path,name,image:false,storage_path:'/actual/results/'+path,...extra};}
const same=[
 item('trace_app2/down_sampled_swcs_app2','trace_app2/down_sampled_swcs_app2/60575.swc'),
 item('gcut_output/selected_for_pruning','gcut_output/selected_for_pruning/60575.swc'),
 item('gcut_output/selected_resampled_step1','gcut_output/selected_resampled_step1/60575.swc'),
 item('pruning_swcs','pruning_swcs/60575.swc')
];
assert.deepEqual(resources.grouped(same).map(g=>g.id),['app2','gcut','resample','pruning']);
assert.equal(new Set(same.map(a=>resources.describe(a).role)).size,4);
const pair=[item('mip_comparison','baseline/pruning_swcs/60575.swc','v0.83_60575.swc',{scope:'comparison',cell:'60575'}),
 item('mip_comparison','candidate/pruning_swcs/60575.swc','v0.83a_60575.swc'),
 item('mip_comparison','per_case.csv','300例逐例指标.csv')];
assert.deepEqual(resources.grouped(pair).map(g=>g.id),['paired_case','paired_batch']);
assert.equal(resources.describe(pair[0]).role,'v0.83 最终 SWC');
assert.equal(resources.describe(pair[1]).role,'v0.83a 最终 SWC');
const masks=[item('mask_step_mips','mask_step_mips/60575_07_g.json','60575_07_g.json'),
 item('mask_step_mips','mask_step_mips/60575_02_b.json','60575_02_b.json')];
assert.deepEqual(resources.grouped(masks)[0].items.map(a=>a.name),['60575_02_b.json','60575_07_g.json']);
assert.match(resources.describe(masks[0]).role,/受约束延伸/);
assert.match(resources.describe(item('regression_guard','regression_guard/60575.json')).role,/回归保护/);
assert.match(resources.describe(item('gcut_audits','gcut_audits/60575_short_leaves_before.swc')).role,/删除前/);
assert.match(resources.describe(item('gcut_audits','gcut_audits/60575_short_leaves_after.swc')).role,/删除后/);
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.textContent='';}
 append(...xs){this.children.push(...xs);}replaceChildren(...xs){this.children=xs;}
 set innerHTML(_){throw Error('Resource names must be rendered as text');}
}
const doc={createElement:tag=>new Element(tag)},container=new Element('div');
const unsafe=item('mask_step_mips','mask_step_mips/60575_01_a.json','<img src=x onerror=alert(1)>.json');
const image=item('mask_mips','mask_mips/60575.png','60575.png',{image:true});
resources.render(container,[...same,...pair,...masks,unsafe,image],a=>'/api/file?path='+encodeURIComponent(a.path),doc);
assert.equal(container.children.length,7); // mask + four SWC stages + two comparison groups
const anchors=container.children.flatMap(section=>section.children[1].children);
assert.equal(anchors.length,10);
for(const a of anchors){assert.match(a.title,/存储路径：\/actual\/results\//);assert.match(a.href,/^\/api\/file\?path=/);}
assert.equal(anchors.find(a=>a.download===unsafe.name).children[1].textContent,unsafe.name+' ↓');
for(const a of same){
 const link=anchors.find(l=>l.href==='/api/file?path='+encodeURIComponent(a.path));
 assert.equal(link.download,'60575.swc');assert.equal(link.title,resources.tooltip(a));
}
const legacy={path:'pruning_swcs/60575.swc',group:'pruning_swcs'};
assert.match(resources.tooltip(legacy),/相对路径/);
assert.equal(resources.grouped([image]).length,0);
const source=fs.readFileSync(require.resolve('../web_console/static/app.js'),'utf8');
assert.match(source,/NeuronResources\.render\(/);
assert.match(source,/button\.title=caption\.title=img\.title=NeuronResources\.tooltip\(a\)/);
console.log('Resource groups: identical filenames, purpose labels, numeric order, actual-path tooltips, scoped downloads and safe text rendering passed.');
