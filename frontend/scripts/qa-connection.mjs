// Isolated real-server regression: never stops or writes to the user's :7860 app.
import { chromium } from 'playwright';
import ts from 'typescript';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const localPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.AGMEN_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const data = fs.mkdtempSync(path.join(os.tmpdir(), 'agmen-connection-'));
const probe = net.createServer();
await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const log = fs.openSync(path.join(data, 'server.log'), 'a');
let server;
let browser;
let releaseSlowHealth = () => {};
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function waitUntil(fn, message, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    try { if (await fn()) return; } catch { /* waiting for a real restart */ }
    await delay(200);
  }
  throw new Error(message);
}
async function start() {
  server = spawn(python, ['-m', 'uvicorn', 'agmen_api:app', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: root, windowsHide: true, stdio: ['ignore', log, log],
    env: { ...process.env, AGMEN_DATA_DIR: data, AGMEN_PROFILE: 'local', OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2' },
  });
  await waitUntil(async () => (await fetch(`${base}/api/health`)).ok, 'Dedicated server did not start');
}
async function stop() {
  if (!server) return;
  const stopped = new Promise(resolve => server.once('exit', resolve));
  if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true });
  else server.kill('SIGTERM');
  await stopped;
  server = undefined;
  await waitUntil(async () => { try { await fetch(`${base}/api/health`); return false; } catch { return true; } }, 'Dedicated server stayed alive');
}
try {
  await start();
  browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? { channel: 'chrome' } : {}) });
  const page = await browser.newPage({ viewport: { width: 1512, height: 1000 } });
  const jsErrors = [];
  page.on('pageerror', e => jsErrors.push(e.message));
  await page.goto(base);
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  const created = await (await page.request.post(`${base}/api/missions`, { data: {
    name: 'Connection regression', drone_count: 5, steps: 20, controller: 'baseline', obstacle: 'none', scenario: 'clusters',
  } })).json();
  assert.ok(created.id);
  await waitUntil(async () => (await (await fetch(`${base}/api/missions/${created.id}`)).json()).status === 'completed', 'Regression mission did not finish');
  await page.reload();
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await page.locator('.drone-marker').first().waitFor();
  await page.getByLabel('Название миссии', { exact: true }).fill('Draft preserved across reconnect');
  await page.getByLabel('Сценарий', { exact: true }).selectOption('patrol');
  const pause = page.getByRole('button', { name: 'Пауза', exact: true });
  if (await pause.isVisible()) await pause.click();
  const beforeSlowHealth = await page.evaluate(() => ({
    mission: localStorage.getItem('agmen-mission'),
    frame: document.querySelector('.timecode')?.textContent,
    markers: [...document.querySelectorAll('.drone-marker')].map(element => element.getAttribute('transform')),
  }));
  assert.equal(beforeSlowHealth.markers.length, 5);

  // A slow reachable server is a loading state. Hold real health requests long
  // enough for the client's own AbortController to time out; never abort the
  // route as a network failure. Every retry stays held until the test releases it.
  let holdHealth = true, firstHeldAt = 0, slowHealthAttempts = 0;
  const heldHealth = new Set();
  releaseSlowHealth = () => {
    holdHealth = false;
    for (const release of heldHealth) release();
    heldHealth.clear();
  };
  await page.route('**/api/health', async route => {
    slowHealthAttempts += 1;
    if (!firstHeldAt) firstHeldAt = Date.now();
    if (holdHealth) await new Promise(resolve => heldHealth.add(resolve));
    // A request whose client deadline elapsed may already be detached.
    await route.continue().catch(() => {});
  });
  const loading = page.getByRole('status', { name: 'Ожидание ответа сервера', exact: true });
  await waitUntil(() => firstHeldAt > 0 && Date.now() - firstHeldAt > 2800, 'Health request exceeds its 2500 ms client deadline', 15000);
  await loading.waitFor({ state: 'visible', timeout: 15000 });
  assert.equal(await loading.locator('.spin').count() > 0, true, 'Slow-health status includes a loading spinner');
  assert.equal(await page.getByRole('alert').count(), 0, 'A deadline does not produce the red connection alert');
  assert.equal(await page.locator('.engine-status.disconnected').count(), 0, 'A deadline does not mark the local engine offline');
  assert.match(await page.locator('.engine-status').innerText(), /LOCAL ENGINE/);
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), 'Draft preserved across reconnect');
  assert.equal(await page.getByLabel('Сценарий', { exact: true }).inputValue(), 'patrol');
  const duringSlowHealth = await page.evaluate(() => ({
    mission: localStorage.getItem('agmen-mission'),
    frame: document.querySelector('.timecode')?.textContent,
    markers: [...document.querySelectorAll('.drone-marker')].map(element => element.getAttribute('transform')),
  }));
  assert.deepEqual(duringSlowHealth, beforeSlowHealth, 'Saved mission and paused frame survive a slow health request');
  fs.mkdirSync(path.join(root, 'test-results'), { recursive: true });
  const loadingScreenshots = [];
  for (const theme of ['light', 'dark']) {
    if (await page.locator('html').getAttribute('data-theme') !== theme) await page.getByRole('button', { name: 'Сменить тему', exact: true }).click();
    await waitUntil(async () => await page.locator('html').getAttribute('data-theme') === theme, `${theme} loading theme`);
    assert.equal(await loading.isVisible(), true);
    const filename = `connection-slow-health-${theme}.png`;
    await page.screenshot({ path: path.join(root, 'test-results', filename), fullPage: true });
    loadingScreenshots.push(filename);
  }
  const slowHealthElapsed = Date.now() - firstHeldAt;
  releaseSlowHealth();
  await page.unroute('**/api/health');
  await loading.waitFor({ state: 'hidden', timeout: 30000 });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  assert.equal(await page.getByRole('alert').count(), 0);
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), 'Draft preserved across reconnect');
  assert.equal(await page.getByLabel('Сценарий', { exact: true }).inputValue(), 'patrol');
  assert.equal(await page.locator('.drone-marker').count(), 5);

  await stop();
  await page.locator('.engine-status.disconnected').waitFor({ timeout: 15000 });
  await page.getByRole('button', { name: 'Повторить подключение', exact: true }).waitFor({ timeout: 15000 });
  assert.match(await page.getByRole('alert').innerText(), /Нет соединения|не ответил вовремя/);
  assert.match(await page.getByRole('alert').innerText(), /ранее загруженные результаты/);
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), 'Draft preserved across reconnect');
  await start();
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor({ timeout: 30000 });
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), 'Draft preserved across reconnect');
  assert.equal(await page.getByLabel('Сценарий', { exact: true }).inputValue(), 'patrol');
  assert.equal(await page.locator('.drone-marker').count(), 5);
  await waitUntil(async () => await page.getByRole('alert').count() === 0, 'Connection error did not clear');
  // The first bootstrap can fail too; the retry control must work before boot exists.
  await page.route('**/api/health', route => route.abort('connectionrefused'));
  await page.reload();
  await page.getByRole('button', { name: 'Повторить подключение', exact: true }).waitFor({ timeout: 15000 });
  await page.unroute('**/api/health');
  await page.getByRole('button', { name: 'Повторить подключение', exact: true }).click();
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor({ timeout: 30000 });
  await page.locator('.drone-marker').first().waitFor();

  // Exercise the shipped request code directly for transport contracts.
  const source = fs.readFileSync(path.join(root, 'frontend/src/api.ts'), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  const moduleURL = `data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`;
  let sentToken;
  await page.route('**/api/test-auth', route => {
    sentToken = route.request().headers()['x-agmen-token'];
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: 'authenticated blob' });
  });
  assert.equal(await page.evaluate(async url => {
    sessionStorage.setItem('agmen-token', 'regression-private-token');
    const text = await (await (await import(url)).requestBlob('/api/test-auth')).text();
    sessionStorage.removeItem('agmen-token');
    return text;
  }, moduleURL), 'authenticated blob');
  assert.equal(sentToken, 'regression-private-token');
  await page.evaluate(async url => {
    const { CONNECTION_EVENT } = await import(url);
    window.__agmenQAConnectionEvents = 0;
    window.addEventListener(CONNECTION_EVENT, () => { window.__agmenQAConnectionEvents += 1; });
  }, moduleURL);
  const transientEventsBefore = await page.evaluate(() => window.__agmenQAConnectionEvents);
  let getAttempts = 0;
  await page.route('**/api/test-transient', async route => {
    getAttempts++;
    if (getAttempts === 1) await route.abort('connectionreset');
    else await route.fulfill({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  });
  assert.equal(await page.evaluate(async url => (await (await import(url)).api('/api/test-transient')).ok, moduleURL), true);
  assert.equal(getAttempts, 2, 'Transient GET should retry once');
  const transientEventsAfter = await page.evaluate(() => window.__agmenQAConnectionEvents);
  assert.equal(transientEventsAfter, transientEventsBefore, 'A successful GET retry never emits the connection-error event');
  assert.match(await page.locator('.engine-status').innerText(), /LOCAL ENGINE/);
  assert.equal(await page.locator('.engine-status.disconnected').count(), 0, 'A transient read keeps the engine online');
  let postAttempts = 0;
  await page.route('**/api/test-write', async route => { postAttempts++; await route.abort('connectionreset'); });
  const postMessage = await page.evaluate(async url => { try { await (await import(url)).api('/api/test-write', { example: true }); } catch (e) { return e.message; } }, moduleURL);
  assert.equal(postAttempts, 1, 'A failed POST must never replay automatically');
  assert.match(postMessage, /Результат команды неизвестен/);
  await page.route('**/api/test-timeout', async () => {});
  const timeoutResult = await page.evaluate(async url => {
    const start = performance.now();
    try { await (await import(url)).api('/api/test-timeout', undefined, { timeoutMs: 150, retries: 0 }); }
    catch (e) { return { message: e.message, elapsed: performance.now() - start, timedOut: e.timedOut }; }
  }, moduleURL);
  assert.match(timeoutResult.message, /не ответил вовремя/);
  assert.equal(timeoutResult.timedOut, true, 'AbortController deadline is explicitly classified as timedOut');
  assert.ok(timeoutResult.elapsed < 1500, 'Request timeout is bounded');
  for (const status of [401, 403, 422]) {
    await page.route(`**/api/test-status-${status}`, route => route.fulfill({ status, contentType: 'application/json', body: '{"detail":"test detail"}' }));
    const message = await page.evaluate(async ({ url, status }) => { try { await (await import(url)).api(`/api/test-status-${status}`); } catch (e) { return e.message; } }, { url: moduleURL, status });
    assert.match(message, new RegExp(`HTTP ${status}`));
    assert.match(message, status === 401 ? /токен/ : status === 403 ? /запретил/ : /параметры/);
  }
  assert.deepEqual(jsErrors, []);
  const report = { passed: true, isolated_port: port, mission_id: created.id, slow_health: { attempts: slowHealthAttempts, held_ms: slowHealthElapsed, loading_spinner: true, engine_stayed_online: true, no_red_alert: true, mission_and_frame_preserved: true, draft_preserved: true, recovered: true, screenshots: loadingScreenshots }, server_stop_restart: true, initial_boot_failure_recovery: true, draft_preserved: true, mission_restored: true, authenticated_blob: true, get_retry_count: getAttempts, transient_get_connection_events: transientEventsAfter - transientEventsBefore, transient_get_stayed_online: true, post_attempt_count: postAttempts, timeout_ms: timeoutResult.elapsed, timeout_classified: timeoutResult.timedOut, distinct_http_errors: [401, 403, 422], browser_errors: jsErrors };
  fs.mkdirSync(path.join(root, 'test-results'), { recursive: true });
  fs.writeFileSync(path.join(root, 'test-results/connection-qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  releaseSlowHealth();
  if (browser) await browser.close();
  await stop();
  fs.closeSync(log);
  // Preserve isolated logs and artifacts in the temp folder for failure diagnosis.
}
