'use strict';
const assert=require('assert/strict');
const {View,buildURL}=require('../web_console/static/vaa3d.js');
const bundle={image:'/data/中文 & #+%.tif',choices:[
 {path:'/data/final.swc',role:'swc',label:'最终',default:true},
 {path:'/data/原始 mask.tif',role:'mask',label:'原始',default:false},
 {path:'/data/repaired.tif',role:'mask',label:'当前',default:true},
 {path:'/data/old.eswc',role:'swc',label:'旧版',default:false}]};
const chosen=new Set(bundle.choices.map(c=>c.path));
const parsed=new URL(buildURL(bundle,chosen));
assert.equal(parsed.protocol,'vaa3d:');assert.equal(parsed.hostname,'open');assert.equal(parsed.hash,'');
assert.equal(parsed.searchParams.get('image'),bundle.image);
assert.deepEqual(parsed.searchParams.getAll('swc'),['/data/final.swc','/data/old.eswc']);
assert.equal(parsed.searchParams.getAll('mask').length,2);
assert.throws(()=>buildURL({},chosen),/缺少/);
const duplicated={...bundle,choices:[...bundle.choices,bundle.choices[0],{path:bundle.image,role:'mask'}]};
assert.equal(new URL(buildURL(duplicated,new Set([...chosen,bundle.image]))).searchParams.size,5);
class Element{
 constructor(){this.children=[];this.dataset={};}
 append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}
 set innerHTML(_){throw Error('Use textContent');}
}
const els={},doc={querySelector:s=>els[s]??(els[s]=new Element()),createElement:()=>new Element()};
let launched;const view=new View(doc,url=>{launched=url;});
view.update(bundle,'job:42');assert.equal(view.button.disabled,false);
assert.deepEqual([...view.selected],['/data/final.swc','/data/repaired.tif']);
const check=view.options.children[1].children[0];check.checked=true;check.onchange();
view.update(JSON.parse(JSON.stringify(bundle)),'job:42');assert.ok(view.selected.has(check.dataset.path));
// Added outputs and repeated polling retain user choices, including deselection.
view.selected.delete('/data/final.swc');
const updated={...bundle,choices:[...bundle.choices,{path:'/data/extra.swc',role:'swc',label:'new',default:false}]};
view.update(updated,'job:42');assert.ok(view.selected.has(check.dataset.path));assert.ok(!view.selected.has('/data/final.swc'));
view.button.onclick();assert.equal(new URL(launched).searchParams.getAll('mask').length,2);
view.clear();assert.equal(view.button.disabled,true);launched=null;view.button.onclick();assert.equal(launched,null);
view.update(bundle,'job:43');assert.ok(!view.selected.has(check.dataset.path));assert.ok(view.selected.has('/data/final.swc'));
view.update({choices:[],unavailable:'missing image'},'job:44');assert.equal(view.button.disabled,true);assert.equal(view.status.textContent,'missing image');
console.log('Vaa3D: encoded multi-file links, defaults, checkboxes, polling, cell reset, missing image and synchronous handoff passed.');
