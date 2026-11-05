// Real private-server regression. All records live in a disposable isolated directory.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import crypto from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.join(root, 'test-results', 'demo-qa');
fs.mkdirSync(out, { recursive: true });
const data = fs.mkdtempSync(path.join(os.tmpdir(), 'agmen-jury-demo-'));
const privateToken = crypto.randomBytes(32).toString('hex');
const localPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.AGMEN_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const probe = net.createServer();
await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const log = fs.openSync(path.join(data, 'server.log'), 'a');
const fixture = path.join(out, 'demo-terrain-fixture.png');
let server, browser, page;
let demoPhase = false;
const browserErrors = [], apiErrors = [], requests = [], headerReads = [];
const report = { passed: false, isolated_port: port, isolated_data: data, checks: {}, screenshots: [] };
console.log(`Isolated demo QA: pid ${process.pid}, port ${port}, data ${data}`);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, label, timeout = 120000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await fn()) return; await delay(180); }
  throw new Error(`Timed out: ${label}`);
}
async function json(route, body, privateRequest = false) {
  const response = await fetch(base + route, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...(privateRequest ? { 'X-AGMEN-Token': privateToken } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  assert.ok(response.ok, `${route}: ${response.status} ${JSON.stringify(result)}`);
  return result;
}
async function completed(id, prefix = '/demo/api', privateRequest = false) {
  let record;
  await until(async () => {
    record = await json(`${prefix}/missions/${id}`, undefined, privateRequest);
    assert.notEqual(record.status, 'failed', record.error);
    return record.status === 'completed';
  }, `mission ${id}`);
  return record;
}
function waitPost(route) {
  return page.waitForResponse(r => new URL(r.url()).pathname === route && r.request().method() === 'POST', { timeout: 60000 });
}
async function clickPost(route, locator) {
  const promise = waitPost(route);
  await locator.click();
  const response = await promise;
  const body = await response.json();
  assert.ok(response.ok(), `${route}: ${response.status()} ${JSON.stringify(body)}`);
  return body;
}
async function closeToast() {
  const close = page.getByRole('button', { name: 'Закрыть уведомление', exact: true });
  if (await close.isVisible()) await close.click();
}
async function capture(name, mobile = false) {
  await closeToast();
  await page.setViewportSize(mobile ? { width: 390, height: 844 } : { width: 1512, height: 1050 });
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: path.join(out, name), fullPage: true });
  report.screenshots.push(name);
}

try {
  const generated = spawnSync(python, ['-c', `from PIL import Image,ImageDraw
import sys
im=Image.new('RGB',(800,600),(237,232,211)); d=ImageDraw.Draw(im)
for x in range(0,800,80): d.line((x,0,x,600),fill=(210,205,186),width=2)
for y in range(0,600,60): d.line((0,y,800,y),fill=(210,205,186),width=2)
d.rectangle((650,25,690,55),fill=(48,46,35)); im.save(sys.argv[1])`, fixture], { cwd: root, windowsHide: true, encoding: 'utf8' });
  assert.equal(generated.status, 0, generated.stderr);
  server = spawn(python, ['-m', 'uvicorn', 'agmen_api:app', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: root, windowsHide: true, stdio: ['ignore', log, log],
    env: { ...process.env, AGMEN_DATA_DIR: data, AGMEN_PROFILE: 'private', AGMEN_API_TOKEN: privateToken, OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2' },
  });
  await until(async () => { try { return (await fetch(`${base}/demo/api/health`, { signal: AbortSignal.timeout(2000) })).ok; } catch { return false; } }, 'isolated private/demo server startup', 60000);
  assert.equal((await fetch(`${base}/api/bootstrap`)).status, 401);
  const image = `data:image/png;base64,${fs.readFileSync(fixture).toString('base64')}`;
  const privateMap = await json('/api/maps', { name: 'PRIVATE TERRAIN FIXTURE', image_data_url: image, width_m: 80, height_m: 60 }, true);
  const privateMission = await json('/api/missions', { name: 'PRIVATE ARCHIVE FIXTURE', drone_count: 5, steps: 20, controller: 'baseline', scenario: 'distribution', obstacle: 'none', wait_for_arrival: false, map_id: privateMap.id }, true);
  await completed(privateMission.id, '/api', true);
  const otherDemo = await json('/demo/api/missions', { name: 'Other demo visitor', drone_count: 5, steps: 20, controller: 'baseline', scenario: 'distribution', obstacle: 'none', wait_for_arrival: false });
  await completed(otherDemo.id);
  const privateBefore = await json('/api/bootstrap', undefined, true);
  const privateMapsBefore = await json('/api/maps', undefined, true);
  const mainDraft = { map_id: privateMap.id, restricted_zones: [{ kind: 'circle', x: .7, y: .4, radius: .03 }], workspace: { width_m: 80, height_m: 60 } };
  const privateStorage = {
    'agmen-theme': 'dark', 'agmen-pro': 'true', 'agmen-mission': privateMission.id,
    'agmen-territory': JSON.stringify(mainDraft),
    [`agmen-destination-${privateMission.id}`]: JSON.stringify(mainDraft),
    [`agmen-stage-plan-${privateMission.id}`]: JSON.stringify({ scenario: 'patrol', origin: { x: 12, y: 9 } }),
  };
  browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? { channel: 'chrome' } : {}) });
  const context = await browser.newContext({ viewport: { width: 1512, height: 1050 }, acceptDownloads: true });
  await context.addCookies([{ name: 'agmen-private-fixture', value: 'must-not-be-sent-to-demo-api', url: base }]);
  page = await context.newPage();
  page.on('pageerror', e => browserErrors.push(e.message));
  page.on('request', request => {
    const pathname = new URL(request.url()).pathname;
    if (!demoPhase || !/^(?:\/demo)?\/api\//.test(pathname)) return;
    const entry = { method: request.method(), path: pathname, headers: null };
    requests.push(entry);
    headerReads.push(request.allHeaders().then(headers => { entry.headers = headers; }));
  });
  page.on('response', response => {
    if (demoPhase && /^(?:\/demo)?\/api\//.test(new URL(response.url()).pathname) && response.status() >= 400) apiErrors.push(`${response.status()} ${response.url()}`);
  });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByRole('alert').filter({ hasText: 'токен' }).waitFor();
  await page.getByRole('button', { name: 'Настройки', exact: true }).click();
  assert.equal(await page.getByLabel('Токен частного сервера', { exact: true }).count(), 1);
  await page.evaluate(({ privateStorage, privateToken }) => {
    for (const [key, value] of Object.entries(privateStorage)) localStorage.setItem(key, value);
    sessionStorage.setItem('agmen-token', privateToken);
  }, { privateStorage, privateToken });
  demoPhase = true;
  await page.goto(base + '/demo', { waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'DEMO ENGINE' }).waitFor();
  assert.match(new URL(page.url()).pathname, /^\/demo\/?$/);
  assert.equal(await page.locator('.demo-badge').innerText(), 'Демо для жюри');
  assert.equal(await page.getByLabel('Количество аппаратов', { exact: true }).inputValue(), '12');
  assert.equal(await page.getByLabel('Длительность расчёта', { exact: true }).inputValue(), '80');
  assert.equal(await page.locator('.drone-marker').count(), 0, 'A new demo visitor must not open another visitor’s mission');
  assert.equal(await page.locator('.map-background').count(), 0, 'Private territory draft is not read by demo');
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'light', 'Demo does not read private theme');
  assert.equal(await page.getByRole('button', { name: 'AI лаборатория', exact: true }).count(), 0);
  assert.equal(await page.getByRole('button', { name: 'Basic', exact: true }).getAttribute('class'), 'selected');
  const boot = await json('/demo/api/bootstrap');
  assert.equal(boot.profile.demo, true); assert.equal(boot.profile.training_allowed, false);
  assert.ok(!boot.missions.some(m => m.id === privateMission.id));
  assert.equal((await fetch(`${base}/demo/api/missions/${privateMission.id}`)).status, 404);
  assert.equal((await fetch(`${base}/demo/api/maps/${privateMap.id}/image`)).status, 404);
  report.checks.private_gate_and_empty_demo_draft = true;
  await capture('demo-first-visit.png');

  await page.getByRole('button', { name: 'Настройки', exact: true }).click();
  assert.equal(await page.getByLabel('Токен частного сервера', { exact: true }).count(), 0);
  await page.getByRole('button', { name: 'Тёмная', exact: true }).click();
  assert.equal(await page.evaluate(() => localStorage.getItem('agmen-demo-theme')), 'dark');
  await page.getByRole('button', { name: 'Миссии', exact: true }).click();
  const editor = page.locator('.map-panel .territory-panel').first();
  const uploadPromise = waitPost('/demo/api/maps');
  await editor.getByLabel('Файл карты территории', { exact: true }).setInputFiles(fixture);
  const upload = await uploadPromise;
  const demoMap = await upload.json(); assert.equal(upload.status(), 201);
  assert.ok(demoMap.image_url.startsWith('/demo/api/maps/'));
  await page.locator('.map-panel .map-background').waitFor();
  assert.match(await page.locator('.map-panel .map-background').getAttribute('href'), /^blob:/);
  const created = await clickPost('/demo/api/missions', page.getByRole('button', { name: 'Рассчитать миссию', exact: true }));
  const mission = await completed(created.id);
  assert.equal(mission.config.drone_count, 12); assert.equal(mission.config.steps, 80); assert.equal(mission.config.map_id, demoMap.id);
  assert.ok(mission.frames.length > 1); assert.equal(mission.arrival.all_arrived, true, 'The actual small demo flock arrives');
  await page.locator('.drone-marker').first().waitFor();
  await capture('demo-mission-desktop.png');
  await capture('demo-mission-mobile.png', true);
  assert.equal(await page.locator('.brand span').innerText(), 'ДЕМО ДЛЯ ЖЮРИ');
  assert.equal(await page.locator('.brand span').isVisible(), true, 'The jury demo label remains visible on a phone');
  const mobile = await page.evaluate(() => ({ width: innerWidth, document: document.documentElement.scrollWidth }));
  assert.ok(mobile.document <= mobile.width + 1);
  const mapBounds = await page.locator('.mission-map').evaluate(svg => { const a = svg.getBoundingClientRect(), b = svg.parentElement.getBoundingClientRect(); return { left: a.left, right: a.right, parentLeft: b.left, parentRight: b.right }; });
  assert.ok(mapBounds.left >= mapBounds.parentLeft - 1 && mapBounds.right <= mapBounds.parentRight + 1);
  report.checks.mission_map_and_mobile = { id: mission.id, frames: mission.frames.length, arrived: mission.arrival.arrived_count, map_id: demoMap.id, mobile, mapBounds };

  await page.setViewportSize({ width: 1512, height: 1050 });
  const arrival = page.getByRole('region', { name: 'После прибытия', exact: true });
  await arrival.getByRole('button', { name: 'Подготовить следующий этап', exact: true }).click();
  const jump = arrival.getByRole('button', { name: 'Перейти к прибытию', exact: true });
  if (await jump.isVisible()) await jump.click();
  const next = await clickPost(`/demo/api/missions/${mission.id}/next-stage`, arrival.getByRole('button', { name: 'Начать следующий этап', exact: true }));
  const nextCompleted = await completed(next.id);
  assert.equal(nextCompleted.stage.parent_mission_id, mission.id);
  assert.ok(await page.evaluate(id => localStorage.getItem(`agmen-demo-destination-${id}`) !== null, mission.id));
  assert.ok(await page.evaluate(id => localStorage.getItem(`agmen-demo-stage-plan-${id}`) !== null, mission.id));
  report.checks.arrival_and_next_stage = { source_id: mission.id, next_id: next.id, status: nextCompleted.status };

  await page.getByRole('button', { name: 'Аналитика', exact: true }).click();
  const exportPromise = page.waitForResponse(r => new URL(r.url()).pathname === `/demo/api/missions/${next.id}/export`);
  const filePromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'JSON', exact: true }).click();
  const exported = await exportPromise; assert.equal(exported.status(), 200);
  const file = await filePromise; await file.saveAs(path.join(out, 'demo-mission-export.json'));
  const exportBody = JSON.parse(fs.readFileSync(path.join(out, 'demo-mission-export.json'), 'utf8')); assert.equal(exportBody.id, next.id);
  await page.getByRole('button', { name: 'AGMEN EDGE', exact: true }).click();
  const pkg = await clickPost('/demo/api/edge/packages', page.getByRole('button', { name: 'Подготовить автономную миссию', exact: true }));
  const device = await clickPost('/demo/api/edge/devices', page.locator('.package-list').getByRole('button', { name: 'Загрузить', exact: true }).first());
  await clickPost(`/demo/api/edge/devices/${device.id}/run`, page.getByRole('button', { name: 'Выполнить 20 шагов', exact: true }));
  await until(async () => (await json(`/demo/api/edge/devices/${device.id}`)).job_status === 'idle', 'first EDGE segment');
  await until(async () => await page.getByRole('button', { name: 'Потеря интернета', exact: true }).isEnabled(), 'EDGE controls unlocked');
  await clickPost(`/demo/api/edge/devices/${device.id}/connectivity`, page.getByRole('button', { name: 'Потеря интернета', exact: true }));
  await clickPost(`/demo/api/edge/devices/${device.id}/run`, page.getByRole('button', { name: 'Выполнить 20 шагов', exact: true }));
  await until(async () => (await json(`/demo/api/edge/devices/${device.id}`)).job_status === 'idle', 'offline EDGE segment');
  const offline = await json(`/demo/api/edge/devices/${device.id}`);
  assert.equal(offline.mode, 'OFFLINE'); assert.ok(offline.history.some(row => row.mode === 'OFFLINE'));
  await until(async () => await page.getByRole('button', { name: 'Синхронизировать', exact: true }).isEnabled(), 'sync unlocked');
  await clickPost(`/demo/api/edge/devices/${device.id}/sync`, page.getByRole('button', { name: 'Синхронизировать', exact: true }));
  const packageFile = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Скачать автономный пакет', exact: true }).first().click();
  await (await packageFile).saveAs(path.join(out, 'demo-mission-package.zip'));
  await capture('demo-edge-desktop.png');
  report.checks.edge_and_downloads = { package_id: pkg.id, device_id: device.id, offline_steps: offline.step, journal_entries: offline.journal_entries };

  const savedDemo = await page.evaluate(() => localStorage.getItem('agmen-demo-mission'));
  await page.goto(base + '/demo/', { waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'DEMO ENGINE' }).waitFor();
  await page.locator('.drone-marker').first().waitFor();
  assert.equal(await page.evaluate(() => localStorage.getItem('agmen-demo-mission')), savedDemo);
  const privateAfterStorage = await page.evaluate(keys => Object.fromEntries(keys.map(key => [key, localStorage.getItem(key)])), Object.keys(privateStorage));
  assert.deepEqual(privateAfterStorage, privateStorage);
  assert.equal(await page.evaluate(() => sessionStorage.getItem('agmen-token')), privateToken);
  const privateAfter = await json('/api/bootstrap', undefined, true);
  const privateMapsAfter = await json('/api/maps', undefined, true);
  assert.deepEqual(privateAfter.missions.map(m => m.id), privateBefore.missions.map(m => m.id));
  assert.deepEqual(privateMapsAfter.map(m => m.id), privateMapsBefore.map(m => m.id));
  await Promise.all(headerReads);
  assert.ok(requests.length > 15);
  for (const request of requests) {
    assert.ok(request.path.startsWith('/demo/api/'), `Private API leak: ${request.path}`);
    assert.equal(request.headers['x-agmen-token'], undefined, 'Demo never sends the private token');
    assert.equal(request.headers.cookie, undefined, 'Demo API omits private same-origin cookies');
    assert.ok(!request.path.includes('/demo/demo/') && !request.path.includes('/demo/api/demo/'), 'Namespace is applied once');
  }
  assert.ok(requests.some(r => r.path.endsWith('/image')));
  assert.ok(requests.some(r => r.path.endsWith('/export')));
  assert.ok(requests.some(r => r.path.endsWith('/download')));
  assert.deepEqual(browserErrors, []); assert.deepEqual(apiErrors, []);
  report.checks.isolation = { private_api_still_requires_token: (await fetch(`${base}/api/bootstrap`)).status === 401, private_archives_unchanged: true, private_browser_storage_unchanged: true, demo_saved_mission_restored: true, no_auth_or_cookie_headers: true, scoped_api_requests: requests.length };
  report.requests = requests.map(({ method, path }) => ({ method, path }));
  report.browser_errors = browserErrors; report.api_errors = apiErrors; report.passed = true;
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  report.error = error.stack; report.browser_errors = browserErrors; report.api_errors = apiErrors;
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  if (page) { await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {}); console.error(await page.locator('body').innerText().catch(() => '')); }
  console.error(fs.readFileSync(path.join(data, 'server.log'), 'utf8').slice(-6000));
  throw error;
} finally {
  await browser?.close();
  if (server && server.exitCode === null) {
    const stopped = new Promise(resolve => server.once('exit', resolve));
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true });
    else server.kill('SIGTERM');
    await stopped;
  }
  fs.closeSync(log);
}
