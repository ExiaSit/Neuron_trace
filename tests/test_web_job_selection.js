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

// Exercise real selection/poll functions with deferred network responses.
context.calls=[];context.replacements=[];
context.history.replaceState=(_a,_b,url)=>context.replacements.push(String(url));
vm.runInContext(`showArtifacts=()=>{};showComparison=()=>{};renderJobs=async()=>{};`,context);
const job=(id,cell)=>({id,name:id,request:{ids:[cell],start:7,stop:6},method_version:'v0.83a',progress:{status:'complete',completed:{}},artifacts:[],log:'',note:''});
function defer(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
(async()=>{
 const loading=defer();
 context.handler=path=>{context.calls.push(path);return path.includes('&cell=')?Promise.resolve(job('new','27721')):loading.promise;};
 vm.runInContext(`api=path=>handler(path);selected='old';currentJob={id:'old',request:{ids:['43199']}};$('#cell').value='43199';`,context);
 const selection=vm.runInContext(`selectJob('new')`,context);
 await vm.runInContext('updateJob()',context);
 assert.equal(context.calls.length,1,'no polling before new task finishes loading');
 assert.equal(els['#cell'].value,'');
 loading.resolve(job('new','27721'));await selection;
 assert.ok(context.calls.includes('/api/job?id=new&cell=27721'));
 assert.ok(!context.calls.some(x=>x.includes('cell=43199')));
 assert.equal(els['#cell'].value,'27721');
 // Even a stale DOM value must never be sent to a different task.
 const n=context.calls.length;els['#cell'].value='43199';await vm.runInContext('updateJob()',context);assert.equal(context.calls.length,n);
 // Errors from an old task must not overwrite a newly selected task.
 els['#cell'].value='27721';const polling=defer();context.handler=()=>polling.promise;
 const oldPoll=vm.runInContext('updateJob()',context);
 vm.runInContext(`selected='other';currentJob=null;`,context);
 polling.reject(Error('样本不属于本任务'));await oldPoll;
 assert.equal(vm.runInContext('pollBusy',context),false);
 // Deleted links stop polling and clear only job/cell URL state.
 context.handler=()=>Promise.reject(Error('任务不存在'));
 await vm.runInContext(`selectJob('deleted','43199')`,context);
 assert.equal(vm.runInContext('selected',context),null);
 assert.equal(vm.runInContext('currentJob',context),null);
 assert.match(els['#notice'].textContent,/已删除或不存在/);
 assert.equal(els['#editor'].hidden,false);assert.equal(els['#results'].hidden,true);
 assert.ok(context.replacements.length>0);
 // A prior error response arriving late must not clear the newer selection.
 const first=defer(),second=defer();context.handler=path=>path.includes('first')?first.promise:second.promise;
 const one=vm.runInContext(`selectJob('first')`,context);const two=vm.runInContext(`selectJob('second')`,context);
 first.reject(Error('任务不存在'));await one;
 assert.equal(vm.runInContext('selected',context),'second');
 context.handler=()=>Promise.resolve(job('second','43199'));second.resolve(job('second','43199'));await two;
 assert.equal(vm.runInContext('currentJob.id',context),'second');assert.equal(els['#notice'].hidden,true);
 // Batch status filters include all concurrent workers and timeout failures.
 context.samples=[{id:'20',status:'running',stage_name:'G-Cut'},{id:'3',status:'complete'},
 {id:'21',status:'running',stage_name:'Mask d'},{id:'4',status:'timeout'},
 {id:'5',status:'pending'},{id:'6',status:'interrupted'}];
 const ids=(filter,search='')=>JSON.parse(vm.runInContext(`JSON.stringify(batchFiltered(samples,${JSON.stringify(filter)},${JSON.stringify(search)}).map(s=>s.id))`,context));
 assert.deepEqual(ids('running'),['20','21']);assert.deepEqual(ids('failed'),['4']);
 assert.deepEqual(ids('unfinished'),['4','5','6','20','21']);assert.deepEqual(ids('all','2'),['20','21']);
 vm.runInContext(`currentJob={id:'batch',request:{ids:samples.map(s=>s.id)},sample_status:{samples}};renderBatchStatus(currentJob);`,context);
 els['#batch-status-counts'].children[2].onclick();
 assert.equal(els['#batch-status-list'].children.length,2);
 let changed=0;els['#cell'].onchange=()=>changed++;
 els['#batch-status-list'].children[1].onclick();assert.equal(els['#cell'].value,'21');assert.equal(changed,1);
 els['#batch-status-search'].value='20';els['#batch-status-search'].oninput();
 assert.equal(els['#batch-status-list'].children.length,1);
 console.log('Job selection: loading race, stale sample, stale error, deleted link and overlapping selections passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
