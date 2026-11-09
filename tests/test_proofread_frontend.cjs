/* Dependency-free browser-state regression tests. Run: node tests/test_proofread_frontend.cjs */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../tools/proofread/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
let modalChoice = 'cancel';
class Element {
  constructor(id='') {
    this.id=id; this.value=''; this.textContent=''; this.dataset={}; this.children=[];
    this.disabled=false; this.hidden=false; this.open=false; this.scrollTop=0; this.listeners={}; this.attributes={};
    this.classList={toggle() {}}; this.paused=true; this.playCalls=0;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  get childElementCount() { return this.children.length; }
  setAttribute(key,value) { this.attributes[key]=value; }
  getAttribute(key) { return this.attributes[key]; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(type,handler) { this.listeners[type]=handler; }
  removeEventListener(type,handler) { if(this.listeners[type]===handler)delete this.listeners[type]; }
  pause() { this.paused=true; }
  load() {}
  play() { this.paused=false; this.playCalls++; return Promise.resolve(); }
  showModal() {
    this.open=true;
    queueMicrotask(()=>this.listeners.click({target:{closest:()=>({dataset:{choice:modalChoice}})}}));
  }
  close() { this.open=false; }
  click() { return this.onclick?.(); }
}
for(const match of html.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element(match[1]));
elements.get('speed').value='1';
const filters=['all','pending','done'].map(value=>{const el=new Element();el.dataset.filter=value;return el;});
const events={}, messages=[], requests=[];
const origin='http://192.168.50.23:7860';
const parent={postMessage:(data,to)=>{assert.equal(to,origin);messages.push(data);}};
const session={token:'test-token',directory:'',project:'',language:'ZH',role:'training',last_id:null,items:[]};
let disk={id:'clip-1',name:'sub/中文.wav',text:'原始文本',version:'v1',reviewed:false};
let asr={available:true,message:'ready',job:{running:false,total:0,completed:0,succeeded:0,skipped:0,failed:0}};
let manifest={stale:false,message:'ready',manifest:'manifest.jsonl',training_running:false};
const validationManifest={stale:true,status:'missing',manifest:'',training_running:false};
const activeManifest=()=>session.role==='validation'?validationManifest:manifest;
let failSave=false;
function snapshot() {
  return {...session,items:session.directory?[{id:disk.id,name:disk.name,reviewed:disk.reviewed,missing_text:!disk.text.trim()}]:[],asr:structuredClone(asr)};
}
const sandbox={
  console,URL,URLSearchParams,Promise,queueMicrotask,location:{origin},confirm:()=>true,setInterval:()=>0,
  document:{documentElement:{},getElementById:id=>elements.get(id),createElement:()=>new Element(),
    querySelectorAll:selector=>selector==='[data-filter]'?filters:[],addEventListener:(type,handler)=>events[`document:${type}`]=handler},
  window:{parent,addEventListener:(type,handler)=>events[type]=handler},
  fetch:async(url,options)=>{
    const [route,query]=url.slice(4).split('?');
    const body=options.body===undefined?undefined:JSON.parse(options.body);
    requests.push({route,body,headers:options.headers});
    if(!(route==='session'&&options.method==='POST'))assert.equal(options.headers['X-Session-Token'],session.token);
    let data;
    if(route==='session')data=snapshot();
    else if(route==='select'){Object.assign(session,body);data=snapshot();}
    else if(route==='item'){assert.equal(new URLSearchParams(query).get('id'),disk.id);data={...disk};}
    else if(route==='position')data={};
    else if(route==='manifest/status')data={...activeManifest(),role:session.role};
    else if(route==='manifest/update'){
      assert.equal(body.role,session.role);
      const dataset=activeManifest();
      Object.assign(dataset,{stale:false,status:'ready',manifest:session.role==='validation'?'validation.jsonl':'manifest.jsonl',stats:{samples:3,valid:2,missing_text:1}});
      data={status:{...dataset,role:session.role},meta:{manifest:manifest.manifest}};
    }
    else if(route==='asr/status')data=structuredClone(asr);
    else if(route==='asr/start'){asr.job={...asr.job,running:true,total:1};data=structuredClone(asr);}
    else if(route==='asr/stop'){asr.job.running=false;asr.job.cancelled=true;data=structuredClone(asr);}
    else if(route==='asr/item')data={id:disk.id,version:disk.version,text:'重新识别结果'};
    else if(route==='save'){
      if(failSave)return{ok:false,json:async()=>({error:'TXT changed on disk'})};
      disk={...disk,text:body.text,version:'v2',reviewed:body.reviewed};activeManifest().stale=true;data={...disk};
    } else if(route==='review'){disk.reviewed=body.reviewed;data={...disk};}
    else throw new Error(`Unexpected API route ${route}`);
    return{ok:true,json:async()=>data};
  }
};
vm.createContext(sandbox);
vm.runInContext(script,sandbox,{filename:'proofread-inline.js'});
const evaluate=expression=>vm.runInContext(expression,sandbox);
const tick=()=>new Promise(resolve=>setImmediate(resolve));

(async()=>{
  await tick();
  assert.equal(evaluate('ready'),true);
  assert(messages.some(message=>message.type==='proofread-ready'));
  await evaluate("run(()=>changeContext({directory:'D:/音频',project:'voice',language:'ZH',ui_language:'en_US'},true))");
  assert.equal(elements.get('editor').value,'原始文本');
  assert.equal(elements.get('audio').playCalls,0,'Selecting a clip must not autoplay');
  assert.equal(evaluate('locale'),'en_US');
  assert.equal(evaluate('context.role'),'training');
  assert.equal(elements.get('manifestUpdate').textContent,'Update training manifest');

  elements.get('editor').value='未保存的人工修改';
  await evaluate('poll()');
  assert.equal(elements.get('editor').value,'未保存的人工修改','Polling must preserve dirty text');
  modalChoice='cancel';
  await evaluate("run(()=>changeContext({directory:'D:/other',project:'other',language:'ZH'},true))");
  assert.equal(evaluate('context.project'),'voice');
  assert.equal(elements.get('editor').value,'未保存的人工修改');
  assert(messages.some(message=>message.type==='proofread-context-rejected'&&message.project==='voice'));
  const beforeRoleSwitch=requests.filter(request=>request.route==='select').length;
  await evaluate("run(()=>changeContext({...context,role:'validation'},true))");
  assert.equal(evaluate('context.role'),'training');
  assert.equal(requests.filter(request=>request.route==='select').length,beforeRoleSwitch);
  assert.equal(messages.at(-1).role,'training','A rejected role change must restore the original role');

  failSave=true;
  await elements.get('save').click();
  assert.equal(elements.get('editor').value,'未保存的人工修改','Save conflicts must preserve editor text');
  assert.equal(disk.text,'原始文本');
  assert.equal(elements.get('message').textContent,'TXT changed on disk');
  failSave=false;
  await elements.get('save').click();
  assert.equal(disk.text,'未保存的人工修改');
  assert.equal(manifest.stale,true);
  assert.equal(evaluate('dirty()'),false);

  const saveCount=requests.filter(request=>request.route==='save').length;
  await elements.get('retranscribe').click();
  assert.equal(elements.get('editor').value,'重新识别结果');
  assert.equal(evaluate('dirty()'),true);
  assert.equal(disk.text,'未保存的人工修改','Single ASR must not save automatically');
  assert.equal(requests.filter(request=>request.route==='save').length,saveCount);

  modalChoice='discard';
  await elements.get('batchStart').click();
  assert.equal(asr.job.running,true);
  assert.equal(elements.get('editor').value,disk.text,'Discarded edits must not survive batch start');
  assert.equal(elements.get('directoryInput').disabled,true);
  const selectCount=requests.filter(request=>request.route==='select').length;
  await evaluate("run(()=>changeContext({directory:'D:/other',project:'other',language:'ZH'},true))");
  assert.equal(requests.filter(request=>request.route==='select').length,selectCount,'Batch jobs must lock context switches');
  await evaluate("run(()=>changeContext({...context,role:'validation'},true))");
  assert.equal(requests.filter(request=>request.route==='select').length,selectCount,'Batch jobs must lock role switches');
  await elements.get('batchStop').click();
  assert.equal(asr.job.running,false);

  await elements.get('manifestUpdate').click();
  assert.equal(manifest.stale,false);
  assert(messages.some(message=>message.type==='proofread-updated'&&message.manifest==='manifest.jsonl'));
  manifest.training_running=true;
  await evaluate('refreshManifest()');
  assert.equal(elements.get('manifestUpdate').disabled,true);
  manifest.training_running=false;

  await evaluate("run(()=>changeContext({...context,role:'validation'},true))");
  assert.equal(evaluate('context.role'),'validation','Role-only changes must reselect the context');
  assert.equal(requests.filter(request=>request.route==='select').at(-1).body.role,'validation');
  assert.equal(elements.get('projectLabel').textContent,'Project: voice · Validation set');
  assert.equal(elements.get('manifestUpdate').textContent,'Update validation manifest');
  assert.equal(elements.get('manifestBadge').textContent,'Update needed');
  await elements.get('manifestUpdate').click();
  assert.equal(elements.get('manifestPath').textContent,'validation.jsonl');
  assert.match(elements.get('manifestStats').textContent,/2 valid, 1 filtered out, 1 missing text/);
  assert(messages.some(message=>message.type==='proofread-updated'&&message.role==='validation'&&message.manifest==='validation.jsonl'));
  await evaluate("run(()=>changeContext({...context,role:'training',ui_language:'zh_CN'},true))");
  assert.equal(elements.get('manifestUpdate').textContent,'更新训练清单');
  assert.equal(elements.get('manifestPath').textContent,'manifest.jsonl');
  await evaluate("run(()=>changeContext({...context,ui_language:'en_US'},true))");

  events.message({origin:'http://untrusted.example',source:parent,data:{type:'proofread-context',directory:'D:/untrusted'}});
  assert.equal(evaluate('pendingContext'),null,'Messages from a different origin must be ignored');
  assert.equal(evaluate("t('saved',{count:1})"),'Saved');
  console.log('Proofreading frontend: workflow and validation role scenarios passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
