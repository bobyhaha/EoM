const fs = require('node:fs');
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.env.JSDOM_MODULE || 'jsdom');

async function main() {
  const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const errors = [], scrollTargets = [];
  let fetchRequests = 0;
  const console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error.message));
  const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), {
    runScripts: 'dangerously',
    url: 'http://localhost/',
    virtualConsole: console,
    beforeParse(window) {
      window.fetch = async () => { fetchRequests += 1; return {ok: true, json: async () => data}; };
      window.setInterval = () => 0;
      window.HTMLElement.prototype.scrollIntoView = function () { scrollTargets.push(this.id); };
    },
  });
  await new Promise(resolve => setImmediate(resolve));
  const $ = id => dom.window.document.getElementById(id);
  assert.match($('updated').textContent, /^Updated /);
  let checked = 0;
  for (const [panel, pairs] of [['comparison', data.paired], ['repeated', data.repeated]]) {
    const buttons = [...$(panel).querySelectorAll('.score-link')];
    const expected = pairs.flatMap(pair => pair.tasks.flatMap(task => ['original', 'teams'].map(arm => ({pair, task, arm}))));
    assert.equal(buttons.length, expected.length);
    for (const [index, button] of buttons.entries()) {
      const {pair, task, arm} = expected[index];
      const phase = pair.repeat == null ? 'evaluation' : 'replication';
      const row = data.arms[arm].episodes.find(row => row.task_id === task.task_id && row.phase === phase && row.epoch === pair.epoch && (row.repeat ?? null) === (pair.repeat ?? null));
      assert.ok(row);
      button.click();
      assert.equal($('arm').value, arm);
      assert.equal($('artifact').getAttribute('href'), row.artifact + '/');
      assert.equal($('task-prompt').textContent, data.tasks[task.task_id].problem);
      assert.equal(scrollTargets.at(-1), 'episode-panel');
      assert.equal(button.type, 'button');
      assert.ok(button.getAttribute('aria-label').includes(task.task_id));
      if (arm === 'teams') assert.equal($('replay').querySelector('a').getAttribute('href'), row.replay);
      checked += 1;
    }
  }
  const previousArtifact = $('artifact').getAttribute('href');
  await dom.window.refresh();
  assert.equal($('artifact').getAttribute('href'), previousArtifact);
  if ($('snapshot-data')) assert.equal(fetchRequests, 0);
  else assert.ok(fetchRequests > 0);
  assert.deepEqual(errors, []);
  dom.window.close();
  process.stdout.write(`Campaign viewer checks passed: ${checked} score links select the exact arm/task/epoch/repeat; selection survives refresh; ${fetchRequests} data fetches.\n`);
}

main().catch(error => { process.stderr.write(String(error.stack) + '\n'); process.exitCode = 1; });
