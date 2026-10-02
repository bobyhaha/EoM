// DOM regression tests for event playback, state reconstruction and stable reading.
// JSDOM_MODULE may point to an existing jsdom installation.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { JSDOM, VirtualConsole } = require(process.env.JSDOM_MODULE || 'jsdom');
const html = fs.readFileSync(process.argv[2], 'utf8');
const errors = [], intervals = new Map();
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => errors.push(e.message));
let width = 820, resize, timerId = 0;
const dom = new JSDOM(html, {
  runScripts: 'dangerously', url: 'file:///replay.html', virtualConsole,
  beforeParse(window) {
    window.ResizeObserver = class { constructor(fn) { resize = fn; } observe() {} };
    window.setInterval = (fn, delay) => { intervals.set(++timerId, {fn, delay}); return timerId; };
    window.clearInterval = id => intervals.delete(id);
    window.HTMLElement.prototype.scrollIntoView = function() {};
    Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', { get() { return width; } });
  }
});
const document = dom.window.document, $ = id => document.getElementById(id);
const payload = JSON.parse($('replay-data').textContent), round = payload.rounds[0];
const input = (id, value, type='input') => { $(id).value=String(value); $(id).dispatchEvent(new dom.window.Event(type)); };
const seek = n => input('event-slider', n);
assert.ok(round.events.length > 2, 'Fixture must contain intermediate events');
assert.equal($('event-slider').value, '0');
assert.equal($('event-slider').max, String(round.events.length));
assert.equal(document.querySelectorAll('.node').length, payload.initial_roster.length);
assert.equal(document.querySelectorAll('.event').length, round.events.length, 'Full transcript remains readable');
assert.ok(!$('round-outcome').textContent.includes('score'), 'No final outcome at start');
$('next').click();
assert.equal($('event-slider').value, '1', 'Next advances one event, not a whole task');
assert.equal(document.querySelectorAll('.event.current').length, 1);
assert.equal(document.querySelector('.event.current').dataset.event, '1');
$('previous').click();
assert.equal($('event-slider').value, '0');
for (const a of payload.initial_roster) {
  assert.equal(document.querySelector(`[data-agent="${a.name}"] small`).textContent, a.wealth.toFixed(2));
}
// A timer tick advances one event. Reading pauses the timer.
$('play').click();
assert.equal($('play').textContent, 'Pause');
[...intervals.values()][0].fn();
assert.equal($('event-slider').value, '1');
dom.window.dispatchEvent(new dom.window.WheelEvent('wheel'));
assert.equal($('play').textContent, 'Play');
assert.equal(intervals.size, 0);
// Playback must not replace transcript nodes, close disclosures or clear selection.
const article = document.querySelector('.event'), details = article.querySelector('details');
details.open = true;
const textNode = article.querySelector('.message').firstChild;
if (textNode) {
  const range = document.createRange(); range.selectNodeContents(textNode);
  dom.window.getSelection().addRange(range);
}
const selectedText = dom.window.getSelection().toString();
$('next').click();
assert.strictEqual(document.querySelector('.event'), article);
assert.equal(details.open, true);
assert.equal(dom.window.getSelection().toString(), selectedText);
// Jumps and reverse scrubbing reconstruct balances instead of exposing the final snapshot.
const auctionIndex=round.events.findIndex(e=>e.event==='auction');
if (auctionIndex>=0) {
  const auction=round.events[auctionIndex];seek(auctionIndex+1);
  for (const name of auction.members) {
    const total=auction.members.reduce((sum,n)=>sum+auction.contributions[n],0);
    const cost=total?auction.paid*auction.contributions[name]/total:0;
    const expected=auction.opening_wealth[name]-cost+(auction.credits?.[name]||0);
    assert.equal(document.querySelector(`[data-agent="${name}"] small`).textContent, expected.toFixed(2));
  }
}
seek(0);
assert.ok(!$('round-outcome').textContent.includes('score'));
if(payload.interaction_protocol==='rounds') {
  if(payload.round_schedule==='compact') {
    const activation=round.events.find(e=>e.event==='team_activation');
    assert.ok(activation, 'Compact replay records collective act/abstain decisions');
    assert.ok(document.querySelector(`[data-event="${round.events.indexOf(activation)+1}"]`).textContent.includes('yes votes'),
      'Collective decision exposes vote count and threshold');
    assert.ok(!round.events.some(e=>e.event==='finalization_recovery'), 'Compact never forces an unfunded final action');
  }
  const feeIndex=round.events.findIndex(e=>e.event==='coordination_fee');
  if(feeIndex>=0){
    seek(feeIndex);
    const fee=round.events[feeIndex], balances=Object.fromEntries(fee.members.map(name=>[name,Number(document.querySelector(`[data-agent="${name}"] small`).textContent)]));
    seek(feeIndex+1);
    for(const [name,cost]of Object.entries(fee.costs))
      assert.ok(Math.abs(Number(document.querySelector(`[data-agent="${name}"] small`).textContent)-balances[name]+cost)<.011,'Coordination fee is burned exactly once in playback');
  }
  const payoutIndex=round.events.findIndex(e=>e.event==='path_reward'), payout=round.events[payoutIndex];
  assert.ok(payoutIndex>=0);
  seek(payoutIndex);
  const before=Object.fromEntries(payload.initial_roster.map(a=>[a.name,Number(document.querySelector(`[data-agent="${a.name}"] small`).textContent)]));
  seek(payoutIndex+1);
  for(const [name,amount]of Object.entries(payout.credits)) {
    const actual=Number(document.querySelector(`[data-agent="${name}"] small`).textContent);
    assert.ok(Math.abs(actual-before[name]-amount)<0.011,'Path reward credits the historical members exactly once');
  }
  const nextSettlement=round.events.findIndex((e,i)=>i>payoutIndex&&e.event==='settlement');
  seek(nextSettlement+1);
  for(const [name,wealth]of Object.entries(round.events[nextSettlement].wealth))
    assert.equal(document.querySelector(`[data-agent="${name}"] small`).textContent,wealth.toFixed(2));
  assert.match($('events').textContent,/R\/N to each member/);
  const decisions=payload.rounds.reduce((sum,r)=>sum+r.events.filter(e=>e.event==='settlement').length,0);
  assert.equal(document.querySelectorAll('#membership thead th').length,decisions+2+(payload.evolution_enabled?payload.rounds.length:0),'Timeline contains each decision round and evolution boundary');
  seek(0);
  const historicalTeam=round.events.find(e=>e.event==='team_message'&&e.channel==='pre_bid').team;
  // Some traces identify a discussion by group rather than team; its member snapshot is authoritative.
  const firstDiscussion=round.events.find(e=>e.event==='team_message'&&e.channel==='pre_bid');
  const team=historicalTeam||round.events.find(e=>e.event==='membership_committed').membership[firstDiscussion.agent];
  assert.ok([...$('team-filter').options].some(o=>o.value===team),'Dissolved teams remain selectable');
  input('team-filter',team,'change');
  assert.ok($('events').querySelector(`[data-event="${round.events.indexOf(firstDiscussion)+1}"]`),'Historical discussion stays associated with its team');
  const laterDiscussion=round.events.find(e=>e.event==='team_message'&&e.channel==='pre_bid'&&e.step>firstDiscussion.step&&e.agent===firstDiscussion.agent);
  if(laterDiscussion){
    const laterTeam=round.events.find(e=>e.event==='membership_committed'&&e.step===laterDiscussion.step).membership[laterDiscussion.agent];
    if(laterTeam!==team)assert.ok(!$('events').querySelector(`[data-event="${round.events.indexOf(laterDiscussion)+1}"]`),'Switching agents do not attach future messages to past teams');
  }
  input('team-filter','','change');
}
$('events').querySelector('[data-event="2"] .event-jump').click();
assert.equal($('event-slider').value, '2');
// Inspectors and filters still work independently of playback position.
document.querySelector('.node').click();
assert.match($('agent-detail').textContent, /Current strategy/);
$('clear-agent').click();
input('channel-filter','bidding','change');
assert.match($('events').textContent, /pledge|contribution/);
input('channel-filter','team','change');
if(round.events.some(e=>e.event==='finalization_started'))assert.match($('events').textContent,/Final-answer window opened/);
if(round.events.some(e=>e.event==='submission'))assert.match($('events').textContent,/Submitted final answer|Published intermediate step/);
if(round.events.some(e=>e.event==='review_started')) {
  assert.match($('events').textContent,/Independent drafts started/);
  assert.match($('events').textContent,/evidence review/);
  assert.match($('events').textContent,/revised answer/);
  const reviewButton=[...$('phase-nav').querySelectorAll('button')].find(b=>b.textContent.endsWith('Review'));
  assert.ok(reviewButton,'Review phase is directly navigable');
  reviewButton.click();
  assert.equal($('event-slider').value,String(round.events.findIndex(e=>e.event==='review_feedback')+1));
}
input('channel-filter','all','change');
if(round.events.some(e=>e.step===1)) {
  input('step-filter','1','change');
  assert.ok(!$('events').textContent.includes('Step 1 ·'));
  input('step-filter','','change');
}
$('latest').click();
assert.equal($('round-select').value,String(payload.rounds.length-1));
assert.equal($('event-slider').value,String(payload.rounds.at(-1).events.length));
for(const a of payload.rounds.at(-1).agents)assert.equal(document.querySelector(`[data-agent="${a.name}"] small`).textContent,a.wealth.toFixed(2));
$('membership').querySelector('tbody tr td button').click();
assert.equal($('round-label').textContent,'Initial state');
$('play').click();
[...intervals.values()][0].fn();
assert.equal($('round-select').value,'0');
assert.equal($('event-slider').value,'1');
$('play').click();
width=330;resize();
for(const node of document.querySelectorAll('.node'))assert.ok(parseFloat(node.style.left)>0&&parseFloat(node.style.left)<width);
for(let ri=0;ri<payload.rounds.length;ri++){
 const r=payload.rounds[ri], resetIndex=r.events.findIndex(e=>e.event==='evaluation_reset');
 if(resetIndex<0)continue;
 input('round-select',ri,'change');seek(resetIndex+1);
 for(const a of r.events[resetIndex].agents){
  assert.equal(document.querySelector(`[data-agent="${a.name}"] small`).textContent,a.wealth.toFixed(2),'Test reset restores visible wealth');
 }
}
if(payload.evolution_enabled){
 for(let ri=0;ri<payload.rounds.length;ri++){
  const r=payload.rounds[ri];
  input('round-select',ri,'change');
  for(let ei=0;ei<r.events.length;ei++){
   const e=r.events[ei];if(!['agent_born','agent_removed'].includes(e.event))continue;
   seek(ei);assert.equal(Boolean(document.querySelector(`[data-agent="${e.agent}"]`)),e.event==='agent_removed');
   seek(ei+1);assert.equal(Boolean(document.querySelector(`[data-agent="${e.agent}"]`)),e.event==='agent_born');
   if(e.event==='agent_born'){
    document.querySelector(`[data-agent="${e.agent}"]`).click();
    assert.ok(document.getElementById('agent-detail').textContent.includes(e.agent));
    assert.ok(!document.getElementById('agent-detail').innerHTML.includes('NaN'));
   }
  }
 }
}
assert.deepEqual(errors,[]);
dom.window.close();
// A new completed task must not drag a live viewer to the end or rebuild its transcript.
(async()=>{
 const liveData=JSON.parse(JSON.stringify(payload));liveData.status='running';
 const fresh=JSON.parse(JSON.stringify(liveData));fresh.rounds.push({...fresh.rounds.at(-1),round:fresh.rounds.at(-1).round+1});
 const safe=JSON.stringify(liveData).replace(/</g,'\\u003c');
 const liveHtml=html.replace(/(<script id="replay-data" type="application\/json">)[\s\S]*?(<\/script>)/,(_,a,b)=>a+safe+b);
 let poll;
 const live=new JSDOM(liveHtml,{runScripts:'dangerously',url:'http://localhost/replay.html',virtualConsole,beforeParse(w){
  w.ResizeObserver=class{observe(){}};w.fetch=async()=>({ok:true,json:async()=>fresh});
  w.setInterval=(fn,delay)=>{if(delay===2000)poll=fn;return 1};w.clearInterval=()=>{};
  Object.defineProperty(w.HTMLElement.prototype,'clientWidth',{get(){return 820}});
 }});
 const doc=live.window.document;doc.getElementById('next').click();
 const node=doc.querySelector('.event');node.querySelector('details').open=true;
 await poll();
 assert.equal(doc.getElementById('round-select').value,'0');
 assert.equal(doc.getElementById('event-slider').value,'1');
 assert.equal(doc.getElementById('round-select').options.length,fresh.rounds.length+1);
 assert.strictEqual(doc.querySelector('.event'),node);
 assert.equal(node.querySelector('details').open,true);
 assert.deepEqual(errors,[]);live.window.close();
 console.log('Replay DOM checks passed: individual events, balances, seeking, readable transcript, selection, filters, playback, live refresh and narrow layout.');
})().catch(e=>{console.error(e);process.exitCode=1});
