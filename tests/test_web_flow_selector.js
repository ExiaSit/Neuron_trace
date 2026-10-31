'use strict';
const assert=require('assert/strict'),fs=require('fs'),vm=require('vm');
class Element{
 constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.textContent='';this.dataset={};this.classList={toggle(){}};}
 append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;if(this.tag==='select')this.value=x[0]?.value||'';}
 querySelectorAll(){return [];}setAttribute(){}
}
const els={},fields={};for(const k of ['start','stop','cell','log-stage','method-version'])els['#'+k]=new Element('select');
const form=els['#task-form']=new Element('form');
for(const k of ['name','ids','method_version','start','stop','gpu_id'])fields[k]=new Element('input');
fields.method_version=els['#method-version'];fields.start=els['#start'];fields.stop=els['#stop'];
form.elements={namedItem:k=>fields[k],name:fields.name};
const context={document:{querySelector:s=>els[s]??(els[s]=new Element()),querySelectorAll:()=>[],createElement:t=>new Element(t)},
 Option:function(t,v){const o=new Element('option');o.textContent=t;o.value=v;return o;},
 NeuronVaa3D:{View:class{clear(){}update(){}}},
 PruningReasons:{View:class{visibility(){}},CohortView:class{reset(){}load(){}}},
 URL,URLSearchParams,location:new URL('http://localhost:8766/'),history:{replaceState(){}},setInterval(){}};
vm.createContext(context);vm.runInContext(fs.readFileSync('web_console/static/app.js','utf8').replace(/init\(\);\s*$/,''),context);
vm.runInContext(`boot={default_method:'v0.83a',input_defaults:{},stages:Array.from({length:13},(_,i)=>'stage '+i),methods:[
 {id:'v0.81',description:'原版无 Mask 修复',stages:Array.from({length:13},(_,i)=>i===6?'使用合并 mask（不修复）':'old '+i)},
 {id:'v0.82',description:'GSDT / 三维合并',stages:Array.from({length:13},(_,i)=>i===6?'使用合并 mask（不修复）':'gsdt '+i)},
 {id:'v0.83a',description:'多方向 MIP',stages:Array.from({length:13},(_,i)=>'new '+i)}]};`,context);
fields.start.value='9';fields.stop.value='13';fields.name.value='v0.83a 重建实验';fields.method_version.value='v0.82';
vm.runInContext(`$('#method-version').onchange()`,context);
assert.equal(fields.name.value,'v0.82 重建实验');assert.equal(fields.start.value,'9');assert.equal(fields.stop.value,'13');
assert.equal(els['#method-note'].textContent,'GSDT / 三维合并');
assert.match(els['#start'].children[6].textContent,/不修复/);
fields.name.value='我的对照实验';fields.method_version.value='v0.81';vm.runInContext(`$('#method-version').onchange()`,context);
assert.equal(fields.name.value,'我的对照实验');
fields.method_version.value='v0.82';els['#cell'].value='19975';
vm.runInContext(`currentJob={read_only:true,method_version:'v0.81',request:{name:'历史实验',ids:['19975','19976'],start:9,stop:13}};$('#clone').onclick()`,context);
assert.equal(fields.method_version.value,'v0.81');assert.equal(fields.ids.value,'19975');assert.match(els['#start'].children[6].textContent,/不修复/);
vm.runInContext('newTask()',context);
assert.equal(fields.method_version.value,'v0.83a');assert.equal(els['#method-note'].textContent,'多方向 MIP');
assert.ok(fs.readFileSync('web_console/static/index.html','utf8').includes('算法流程版本'));
console.log('Flow selector: version-specific stages, descriptions, custom names, cloned version and new-task default passed.');
