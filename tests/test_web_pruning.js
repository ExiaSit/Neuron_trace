/* Run: node --test tests/test_web_pruning.js */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {parseSWC,model,summaryModel,View}=require('../web_console/static/pruning.js');
const input='1 1 0 0 0 1 -1\n2 3 3 4 0 1 1\n3 3 3 4 12 1 2\n4 3 0 4 12 1 3\n5 3 0 0 1 1 1';
const output='1 1 0 0 0 1 -1\n5 3 0 0 1 1 1';
const report={removed_nodes:{2:{code:'a',reason:'原因 A'},3:{code:'b',reason:'原因 B'}}};
test('3D original parent edges include attachment and classify missing reasons explicitly',()=>{
 const d=model(report,parseSWC(input),parseSWC(output));
 assert.equal(d.removed.size,3);assert.equal(d.missingRecords,1);
 assert.deepEqual(d.groups.map(g=>g.length),[5,12,3]);
 assert.equal(d.groups.reduce((s,g)=>s+g.length,0),20);
 assert.equal(model({removed_nodes:{1:{code:'x'},88:{code:'x'}}},parseSWC(input),parseSWC(output)).unmatched,2);
});
test('no removals, malformed SWC, and aggregate-only historical data',()=>{
 assert.equal(model(report,parseSWC(input),parseSWC(input)).removed.size,0);
 assert.throws(()=>parseSWC('# empty'));assert.throws(()=>parseSWC('1 3 NaN 0 0 1 -1'));
 assert.throws(()=>parseSWC('1 3 0 0 0 1 -1\n1 3 0 0 0 1 -1'));
 const d=summaryModel({removal_counts:{sustained_3d_signal_gap:19,qc_merge_close_point:2}});
 assert.equal(d.groups[0].count,19);assert.equal(d.groups[0].length,null);assert.equal(d.summaryOnly,true);
});
// Small DOM harness exercises actual view/render code without a browser dependency.
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.value='';this.textContent='';}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;if(this.tag==='select')this.value=items[0]?.value||'';}
 setAttribute(k,v){this.attrs[k]=v;}
}
function setup(){
 const ids=['pruning-filter','pruning-reason','pruning-reset','pruning-status','pruning-projections','pruning-stats','result-show-pruning_mips'];
 const elements=Object.fromEntries(ids.map(id=>['#'+id,new Element(id.includes('reason')?'select':'div')]));
 elements['#result-show-pruning_mips'].checked=true;
 global.document={querySelector:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
 global.Option=class extends Element{constructor(text,value){super('option');this.textContent=text;this.value=value;}};
 return elements;
}
function artifacts(prefix=''){
 return [
  {name:'42_pruning_reasons.json',path:prefix+'report',group:'pruning_mips'},
  {name:'42.swc',path:prefix+'input.swc',group:'gcut_output/selected_resampled_step1'},
  {name:'42.swc',path:prefix+'output.swc',group:'pruning_swcs'}];
}
function reply(url){return {ok:true,json:async()=>report,text:async()=>url.endsWith('input.swc')?input:output};}
test('selecting a reason updates all three projections, statistics and reset',async()=>{
 const e=setup();global.fetch=async url=>reply(url);const v=new View();await v.load(artifacts(),a=>a.path);
 assert.equal(e['#pruning-projections'].children.length,3);assert.equal(v.select.children.length,4);
 v.select.value=v.data.groups.find(g=>g.code==='b').key;v.select.onchange();
 assert.match(e['#pruning-status'].textContent,/选中 1 \/ 3/);
 const svg=e['#pruning-projections'].children[1].children[1];
 assert.equal(svg.children[2].attrs.d,'M3,0L3,12'); // XZ for selected node 3 only.
 e['#pruning-reset'].onclick();assert.equal(v.select.value,'all');
 e['#result-show-pruning_mips'].checked=false;v.visibility();assert.equal(v.box.hidden,true);
});
test('historical summaries show counts without invented spatial overlays',async()=>{
 const e=setup();global.fetch=async()=>({ok:true,json:async()=>({removal_counts:{qc_merge_close_point:2}})});
 const v=new View();await v.load([{name:'42_pruning-audit.json',path:'summary',group:'pruning_mips'}],a=>a.path);
 assert.match(e['#pruning-status'].textContent,/仅保留原因汇总/);assert.equal(e['#pruning-projections'].children.length,0);
 assert.equal(v.select.children.length,2);
});
test('a slow previous neuron cannot overwrite the selected neuron',async()=>{
 setup();let release;
 const gate=new Promise(resolve=>{release=resolve;});
 global.fetch=async url=>{if(url.startsWith('old'))await gate;return reply(url);};
 const v=new View(),old=v.load(artifacts('old'),a=>a.path);
 await v.load([{name:'42_combined_process.png',path:'new',group:'pruning_mips'}],a=>a.path);
 release();await old;assert.equal(v.data,null);assert.equal(v.select.disabled,true);
});

test('cohort filtering excludes missing/zero counts, sorts counts, and applies minimum',()=>{
 const {matchingCells}=require('../web_console/static/pruning.js');
 const cells=[{id:'1',status:'recorded',counts:{a:3}},{id:'2',status:'missing',counts:null},{id:'3',status:'recorded',counts:{}},{id:'4',status:'recorded',counts:{a:8,b:1}}];
 assert.deepEqual(matchingCells({cells},'a').map(c=>c.id),['4','1']);
 assert.deepEqual(matchingCells({cells},'a',5).map(c=>c.id),['4']);
 assert.deepEqual(matchingCells({cells},'b',2),[]);
});
function cohortSetup(){
 const elements=setup();for(const id of ['neuron-rule','neuron-rule-min','neuron-rule-status','neuron-rule-matches','neuron-rule-refresh','neuron-rule-clear'])elements['#'+id]=new Element(id==='neuron-rule'?'select':'div');
 return elements;
}
test('cohort selector refresh, click-through, reset and stale response',async()=>{
 const {CohortView}=require('../web_console/static/pruning.js'),elements=cohortSetup();let clicked;
 const data={total:3,recorded:2,missing:1,invalid:0,rules:[{code:'a',reason:'A',cells:1}],cells:[{id:'42',status:'recorded',counts:{a:7}},{id:'43',status:'recorded',counts:{}}]};
 const view=new CohortView(async()=>data,(...args)=>{clicked=args;});
 await view.load('job1');view.select.value='a';view.select.onchange();
 assert.equal(view.matches.children.length,1);view.matches.children[0].onclick();assert.deepEqual(clicked,['42','a']);
 await view.load('job1');assert.equal(view.select.value,'a');assert.equal(view.matches.children.length,1);
 elements['#neuron-rule-clear'].onclick();assert.equal(view.select.value,'');assert.equal(view.matches.children.length,0);
 let release;view.api=()=>new Promise(resolve=>{release=resolve;});const pending=view.load('job2');view.reset();release(data);await pending;
 assert.equal(view.job,null);assert.equal(view.data,null);assert.equal(view.select.disabled,true);
});
