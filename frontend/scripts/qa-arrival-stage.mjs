// Real-browser arrival/stage regression. A temporary server owns every test record.
// Never touches the user's application on :7860 or its persisted maps/missions.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import net from 'node:net';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.join(root, 'test-results', 'arrival-stage-qa');
fs.mkdirSync(out, { recursive: true });
const data = fs.mkdtempSync(path.join(os.tmpdir(), 'agmen-arrival-stage-'));
const localPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.AGMEN_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const originPNG = path.join(out, 'origin-terrain.png');
const destinationPNG = path.join(out, 'destination-terrain.png');
const fixtureID = 'a11a11a11a11';
const probe = net.createServer();
await new Promise(resolve => probe.listen(0, '127.0.0.1', resolve));
const port = probe.address().port;
await new Promise(resolve => probe.close(resolve));
const base = `http://127.0.0.1:${port}`;
const log = fs.openSync(path.join(data, 'server.log'), 'a');
let server, browser, page;
let releaseDelayedPolls = () => {};
const report = { passed: false, isolated_port: port, isolated_data: data, checks: {}, screenshots: [] };
const jsErrors = [], apiErrors = [], transportErrors = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, description, timeout = 60000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    if (await check()) return;
    await delay(150);
  }
  throw new Error(`Timed out: ${description}`);
}
async function get(route) {
  const response = await fetch(base + route);
  const body = await response.json();
  assert.ok(response.ok, `${route}: HTTP ${response.status} ${JSON.stringify(body)}`);
  return body;
}
async function completed(id) {
  let record;
  await until(async () => {
    record = await get(`/api/missions/${id}`);
    assert.notEqual(record.status, 'failed', record.error || 'Mission failed');
    return record.status === 'completed';
  }, `mission ${id} completion`, 120000);
  return record;
}
function waitPost(route) {
  return page.waitForResponse(response => new URL(response.url()).pathname === route && response.request().method() === 'POST', { timeout: 60000 });
}
async function clickPost(route, button) {
  const waiting = waitPost(route);
  await button.click();
  const response = await waiting;
  const body = await response.json();
  assert.ok(response.ok(), `${route}: HTTP ${response.status()} ${JSON.stringify(body)}`);
  return { body, submitted: response.request().postDataJSON() };
}
async function upload(editor, file) {
  const waiting = waitPost('/api/maps');
  await editor.getByLabel('Файл карты территории', { exact: true }).setInputFiles(file);
  const response = await waiting;
  const body = await response.json();
  assert.equal(response.status(), 201, JSON.stringify(body));
  return body;
}
async function closeToast() {
  const close = page.getByRole('button', { name: 'Закрыть уведомление', exact: true });
  if (await close.isVisible()) {
    try { await close.click({ timeout: 1000 }); }
    catch (error) { if (await close.isVisible()) throw error; }
  }
}
async function openEditor(editor) {
  const toggle = editor.locator('.territory-heading > .text-button');
  if (await toggle.getAttribute('aria-expanded') !== 'true') await toggle.click();
}
async function stored(key = 'agmen-territory') {
  return page.evaluate(value => JSON.parse(localStorage.getItem(value) || 'null'), key);
}
async function point(svg, x, y) {
  await svg.scrollIntoViewIfNeeded();
  return svg.evaluate((element, coordinates) => {
    const p = new DOMPoint(44 + coordinates.x * 880, 28 + coordinates.y * 578).matrixTransform(element.getScreenCTM());
    return { x: p.x, y: p.y };
  }, { x, y });
}
async function drawRectangle(editor, svg) {
  await editor.getByRole('button', { name: 'Запретная зона: Прямоугольник', exact: true }).click();
  const from = await point(svg, .8, .05), to = await point(svg, .85, .1);
  await page.mouse.move(from.x, from.y); await page.mouse.down();
  await page.mouse.move(to.x, to.y, { steps: 10 }); await page.mouse.up();
}
function continuity(parent, child, registration) {
  const oldAgents = parent.frames.at(-1).agents;
  const newAgents = child.frames[0].agents;
  assert.equal(newAgents.length, oldAgents.length);
  for (const before of oldAgents) {
    const sourceID = before.source_drone_id ?? before.id;
    const after = newAgents.find(agent => (agent.source_drone_id ?? agent.id) === sourceID);
    assert.ok(after, `Source vehicle ${sourceID} retained`);
    const x = before.x * parent.workspace.width_m - registration.origin_x_m;
    const y = before.y * parent.workspace.height_m - registration.origin_y_m;
    assert.ok(Math.abs(after.x * child.workspace.width_m - x) < 1e-6, `Vehicle ${sourceID}: canonical X continuity`);
    assert.ok(Math.abs(after.y * child.workspace.height_m - y) < 1e-6, `Vehicle ${sourceID}: canonical Y continuity`);
    assert.ok(Math.abs(after.battery - before.battery) < 1e-8, `Vehicle ${sourceID}: battery continuity`);
    assert.equal(after.status, before.status);
    assert.equal(after.link, before.link);
  }
}
async function snapshotBothMaps(theme, width, suffix) {
  await page.setViewportSize({ width, height: width < 600 ? 844 : 1100 });
  await page.evaluate(value => { document.documentElement.dataset.theme = value; localStorage.setItem('agmen-theme', value); }, theme);
  await closeToast();
  const layout = await page.evaluate(() => ({ viewport: innerWidth, scroll: document.documentElement.scrollWidth }));
  assert.ok(layout.scroll <= layout.viewport + 1, `${theme}/${suffix} horizontal overflow: ${JSON.stringify(layout)}`);
  assert.equal(await page.locator('.map-panel .map-background').count(), 1);
  assert.equal(await page.locator('.destination-map-panel .map-background').count(), 1);
  assert.equal(await page.locator('.stage-source-agent').count(), 5);
  const bounds = await page.getByLabel('Карта конечного участка с позициями прибывших аппаратов', { exact: true }).evaluate(element => {
    const rect = element.getBoundingClientRect(), parent = element.parentElement.getBoundingClientRect();
    return { width: rect.width, left: rect.left, right: rect.right, parentLeft: parent.left, parentRight: parent.right };
  });
  assert.ok(bounds.width > 250 && bounds.left >= bounds.parentLeft - 1 && bounds.right <= bounds.parentRight + 1, 'Destination SVG fits its panel');
  const filename = `arrival-${theme}-${suffix}.png`;
  await page.screenshot({ path: path.join(out, filename), fullPage: true });
  report.screenshots.push(filename);
  return { theme, ...layout, bounds };
}

// A real simulator run with a deliberately bounded 20-step budget provides a
// truthful incomplete snapshot. It avoids a 2400-step test job merely to reach
// the public resource cap. All subsequent continuation happens through the API.
const fixtureCode = `from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageDraw
import json, sys
from agmen_simulation import simulate
data, origin, destination, identifier = sys.argv[1:]
for target, bg, caption in [(origin,(237,232,211),'ORIGIN / 120 x 90 m'),(destination,(230,232,215),'DESTINATION / REGISTERED LOCAL MAP')]:
    im=Image.new('RGB',(1200,900),bg); draw=ImageDraw.Draw(im)
    for x in range(0,1200,100): draw.line((x,0,x,900),fill=(205,205,187),width=2)
    for y in range(0,900,100): draw.line((0,y,1200,y),fill=(205,205,187),width=2)
    draw.rectangle((1030,40,1090,80),fill=(47,46,35)); draw.text((30,25),caption,fill=(78,78,60)); im.save(target)
config=dict(name='QA bounded arrival budget',drone_count=5,controller='baseline',model_id='bundled',scenario='distribution',steps=20,seed=42,max_speed=3,safe_distance=1.2,obstacle='none',wait_for_arrival=True,width_m=120,height_m=90,restricted_zones=[],max_completion_steps=20)
result=simulate(config)
assert result['summary']['termination_reason']=='step_limit' and not result['arrival']['all_arrived']
record=dict(id=identifier,kind='mission',name=config['name'],status='completed',config=config,created_at='2000-01-01T00:00:00+00:00',updated_at='2000-01-01T00:00:00+00:00',**result)
folder=Path(data)/'missions'; folder.mkdir(parents=True,exist_ok=True)
(folder/(identifier+'.json')).write_text(json.dumps(record,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(id=identifier,actual_steps=result['summary']['actual_steps'],arrived=result['arrival']['arrived_count'])))`;

try {
  const generated = spawnSync(python, ['-c', fixtureCode, data, originPNG, destinationPNG, fixtureID], {
    cwd: root, windowsHide: true, encoding: 'utf8', env: { ...process.env, OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2' },
  });
  assert.equal(generated.status, 0, generated.stderr || generated.stdout);
  report.fixture = JSON.parse(generated.stdout.trim().split(/\r?\n/).at(-1));
  server = spawn(python, ['-m', 'uvicorn', 'agmen_api:app', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: root, windowsHide: true, stdio: ['ignore', log, log],
    env: { ...process.env, AGMEN_DATA_DIR: data, AGMEN_PROFILE: 'local', OMP_NUM_THREADS: '2', MKL_NUM_THREADS: '2' },
  });
  await until(async () => { try { return (await fetch(`${base}/api/health`)).ok; } catch { return false; } }, 'isolated server startup');
  const originalArchive = fs.readFileSync(path.join(data, 'missions', fixtureID + '.json'), 'utf8');
  browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? { channel: 'chrome' } : {}) });
  page = await browser.newPage({ viewport: { width: 1512, height: 1100 } });
  page.on('pageerror', error => jsErrors.push(error.message));
  page.on('requestfailed', request => {
    const failure = request.failure()?.errorText || '';
    if (new URL(request.url()).pathname.startsWith('/api/') && !failure.includes('ERR_ABORTED')) transportErrors.push(`${request.method()} ${request.url()} ${failure}`);
  });
  page.on('response', response => {
    if (new URL(response.url()).pathname.startsWith('/api/') && response.status() >= 400) apiErrors.push(`${response.status()} ${response.request().method()} ${response.url()}`);
  });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await page.getByRole('button', { name: 'Pro', exact: true }).click();
  const mainPanel = page.locator('.map-panel').first();
  const mainEditor = mainPanel.locator('.territory-panel').first();
  const mainSVG = page.getByLabel('Интерактивное поле симуляции', { exact: true });

  // Clear just the overlays, then reset the whole editor, without deleting data.
  await mainEditor.getByRole('button', { name: 'Новая карта', exact: true }).click();
  await mainEditor.getByLabel('Ширина территории, м', { exact: true }).fill('120');
  await mainEditor.getByLabel('Высота территории, м', { exact: true }).fill('90');
  const firstMap = await upload(mainEditor, originPNG);
  await mainPanel.locator('.map-background').waitFor();
  await drawRectangle(mainEditor, mainSVG);
  await until(async () => (await stored()).restricted_zones.length === 1, 'one manually confirmed zone');
  await mainEditor.getByRole('button', { name: 'Очистить зоны', exact: true }).click();
  await until(async () => (await stored()).restricted_zones.length === 0, 'clear only zones');
  assert.equal((await stored()).map_id, firstMap.id);
  assert.equal((await stored()).workspace.width_m, 120);
  assert.equal(await mainPanel.locator('.terrain-suggestion').count(), 0);
  assert.equal(await mainPanel.locator('.map-background').count(), 1);
  await mainEditor.getByRole('button', { name: 'Новая карта', exact: true }).click();
  await until(async () => (await stored()).map_id === null, 'full editor reset');
  assert.equal((await stored()).workspace.width_m, 80);
  assert.equal((await stored()).workspace.height_m, 60);
  assert.equal(await mainPanel.locator('.map-background').count(), 0);
  assert.equal((await get(`/api/maps/${firstMap.id}`)).id, firstMap.id);
  assert.equal(fs.readFileSync(path.join(data, 'missions', fixtureID + '.json'), 'utf8'), originalArchive);
  await mainEditor.getByLabel('Ширина территории, м', { exact: true }).fill('120');
  await mainEditor.getByLabel('Высота территории, м', { exact: true }).fill('90');
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await openEditor(mainEditor);
  assert.equal(await mainEditor.getByLabel('Ширина территории, м', { exact: true }).inputValue(), '120');
  assert.equal(await mainEditor.getByLabel('Высота территории, м', { exact: true }).inputValue(), '90');
  report.checks.reset = { clear_preserves_map: true, full_reset_80_by_60: true, dimensions_survive_reload: true, archived_map_and_mission_preserved: true };

  const origin = await upload(mainEditor, originPNG);
  await mainPanel.locator('.map-background').waitFor();
  await mainEditor.getByRole('button', { name: 'Очистить зоны', exact: true }).click();
  await page.getByLabel('Название миссии', { exact: true }).fill('Arrival regression source');
  await page.getByLabel('Сценарий', { exact: true }).selectOption('distribution');
  await page.getByLabel('Препятствия', { exact: true }).selectOption('none');
  await page.getByLabel('Длительность расчёта', { exact: true }).selectOption('20');
  await page.getByLabel('Количество аппаратов', { exact: true }).focus();
  await page.getByLabel('Количество аппаратов', { exact: true }).press('Home');
  const waitAll = page.getByLabel('Дожидаться всех аппаратов', { exact: true });
  assert.equal(await waitAll.isChecked(), true, 'Arrival is the default completion criterion');
  const waitControl = await waitAll.evaluate(element => {
    const input = element.getBoundingClientRect(), label = element.closest('label').getBoundingClientRect();
    const text = element.parentElement.querySelector('span').getBoundingClientRect();
    return { width: input.width, height: input.height, textRight: text.right, labelRight: label.right };
  });
  assert.ok(waitControl.width <= 24 && waitControl.height <= 24 && waitControl.textRight <= waitControl.labelRight + 1, `Arrival checkbox and description fit the sidebar: ${JSON.stringify(waitControl)}`);
  const created = await clickPost('/api/missions', page.getByRole('button', { name: 'Рассчитать миссию', exact: true }));
  assert.equal(created.submitted.steps, 20);
  assert.equal(created.submitted.drone_count, 5);
  assert.equal(created.submitted.wait_for_arrival, true);
  assert.equal(created.submitted.map_id, origin.id);
  const parent = await completed(created.body.id);
  assert.equal(parent.arrival.all_arrived, true);
  assert.equal(parent.arrival.arrived_count, 5);
  assert.equal(parent.summary.mission_completed, true);
  assert.ok(parent.summary.actual_steps > 20, 'Real integration extends beyond the nominal 20-step horizon');
  await page.getByRole('button', { name: 'Подготовить следующий этап', exact: true }).waitFor({ state: 'visible', timeout: 30000 });
  await until(async () => await page.getByRole('button', { name: 'Подготовить следующий этап', exact: true }).isEnabled(), 'arrival planner becomes enabled');
  await page.getByRole('button', { name: 'Подготовить следующий этап', exact: true }).click();
  const stage = page.getByRole('region', { name: 'После прибытия', exact: true });
  await until(async () => /5 из 5/.test(await stage.locator('.arrival-intro').innerText()), 'UI arrival progress 5 of 5');
  await stage.getByRole('radiogroup', { name: 'Сценарий после прибытия', exact: true }).waitFor();
  const jump = stage.getByRole('button', { name: 'Перейти к прибытию', exact: true });
  if (await jump.count()) await jump.click();
  report.checks.arrival = { parent_id: parent.id, nominal_steps: 20, actual_steps: parent.summary.actual_steps, arrived: parent.arrival.arrived_count, UI_progress: true };

  // A second PNG has its own dimensions/editor and an explicit metre origin.
  const options = await get(`/api/missions/${parent.id}/next-stage-options`);
  assert.equal(options.allowed, true);
  const region = options.suggested_region;
  const originX = region.origin_x_m - 2, originY = region.origin_y_m - 2;
  const destinationWidth = Math.ceil(region.width_m + 8), destinationHeight = Math.ceil(region.height_m + 8);
  const destinationPanel = stage.locator('.destination-map-panel');
  const destinationEditor = destinationPanel.locator('.territory-panel');
  await openEditor(destinationEditor);
  await destinationEditor.getByLabel('Ширина территории, м', { exact: true }).fill(String(destinationWidth - 2));
  await destinationEditor.getByLabel('Высота территории, м', { exact: true }).fill(String(destinationHeight));
  const detailedMap = await upload(destinationEditor, destinationPNG);
  await destinationPanel.locator('.map-background').waitFor();
  await destinationEditor.getByRole('button', { name: 'Очистить зоны', exact: true }).click();
  await destinationEditor.getByLabel('Ширина территории, м', { exact: true }).fill(String(destinationWidth));
  const scaled = await clickPost(`/api/maps/${detailedMap.id}/calibrate`, destinationEditor.getByRole('button', { name: 'Применить масштаб', exact: true }));
  assert.equal(scaled.body.width_m, destinationWidth);
  await stage.getByLabel('Начало участка X, м', { exact: true }).fill(String(originX));
  await stage.getByLabel('Начало участка Y, м', { exact: true }).fill(String(originY));
  const patrol = options.scenarios.find(value => value.id === 'patrol');
  assert.ok(patrol?.available);
  await stage.getByRole('radio', { name: `После прибытия: ${patrol.label}`, exact: true }).click();
  assert.equal(await stage.getByRole('radio', { name: `После прибытия: ${patrol.label}`, exact: true }).getAttribute('aria-checked'), 'true');
  await until(async () => await stage.getByRole('button', { name: 'Начать следующий этап', exact: true }).isEnabled(), 'stage ready with all actual positions inside registered map');
  const sourceDraft = await stored();
  const destinationDraft = await stored(`agmen-destination-${parent.id}`);
  assert.equal(sourceDraft.map_id, origin.id);
  assert.equal(destinationDraft.map_id, detailedMap.id);
  assert.notEqual(sourceDraft.map_id, destinationDraft.map_id);
  report.layouts = [];
  report.layouts.push(await snapshotBothMaps('light', 1512, 'desktop'));
  report.layouts.push(await snapshotBothMaps('dark', 1512, 'desktop'));
  report.layouts.push(await snapshotBothMaps('light', 390, 'mobile'));
  report.layouts.push(await snapshotBothMaps('dark', 390, 'mobile'));
  await page.setViewportSize({ width: 1512, height: 1100 });
  const childRequest = await clickPost(`/api/missions/${parent.id}/next-stage`, stage.getByRole('button', { name: 'Начать следующий этап', exact: true }));
  assert.equal(childRequest.submitted.map_id, detailedMap.id);
  assert.equal(childRequest.submitted.scenario, 'patrol');
  assert.equal(childRequest.submitted.origin_x_m, originX);
  assert.equal(childRequest.submitted.origin_y_m, originY);
  // Hold the old mission's completed browser poll across an archive selection.
  // Native fetch in completed() bypasses Playwright routing, so the simulation
  // still runs normally and no synthetic mission status is supplied to the app.
  const oldChildRoute = `**/api/missions/${childRequest.body.id}`;
  let holdOldPoll = true;
  const heldPolls = [];
  releaseDelayedPolls = () => { holdOldPoll = false; for (const release of heldPolls.splice(0)) release(); };
  await page.route(oldChildRoute, async route => {
    const response = await route.fetch();
    const record = await response.json();
    if (holdOldPoll && record.status === 'completed') await new Promise(resolve => heldPolls.push(resolve));
    await route.fulfill({ response }).catch(() => {});
  });
  const child = await completed(childRequest.body.id);
  await until(() => heldPolls.length > 0, 'a real completed child poll is held before archive selection', 15000);
  const registration = child.stage.registration;
  assert.equal(child.stage.parent_id, parent.id);
  assert.equal(child.stage.kind, 'next_stage');
  assert.equal(registration.metres_per_metre, 1);
  assert.equal(registration.gps_registered, false);
  assert.equal(child.territory.id, detailedMap.id);
  assert.equal(child.config.scenario, 'patrol');
  continuity(parent, child, registration);
  assert.ok(child.frames.at(-1).agents.some((agent, i) => Math.hypot(agent.x - child.frames[0].agents[i].x, agent.y - child.frames[0].agents[i].y) > .001), 'Child stage contains an actual changed path');
  report.checks.next_stage = { child_id: child.id, parent_id: parent.id, map_id: detailedMap.id, scenario: child.config.scenario, registration, canonical_positions: true, source_ids: true, batteries: true, actual_path: true };

  const incomplete = await get(`/api/missions/${fixtureID}`);
  const continuationOptions = await get(`/api/missions/${fixtureID}/next-stage-options`);
  assert.equal(continuationOptions.can_continue, true);
  assert.equal(continuationOptions.allowed, false);
  await page.getByRole('button', { name: 'Аналитика', exact: true }).click();
  await page.locator('tr').filter({ hasText: incomplete.name }).getByRole('button', { name: 'Открыть', exact: true }).click();
  await page.getByRole('button', { name: 'Миссии', exact: true }).click();
  const continueButton = page.getByRole('button', { name: 'Продолжить до цели', exact: true });
  await continueButton.waitFor();
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), incomplete.name);
  await until(async () => await mainPanel.locator('.map-background').count() === 0, 'archived fixture map replaces the child map');
  const delivered = page.waitForResponse(response => new URL(response.url()).pathname === `/api/missions/${child.id}` && response.request().method() === 'GET');
  releaseDelayedPolls();
  await (await delivered).finished();
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(await page.getByLabel('Название миссии', { exact: true }).inputValue(), incomplete.name, 'A late old poll never replaces the selected archive config');
  assert.equal(await mainPanel.locator('.map-background').count(), 0, 'A late old poll never reattaches the child map');
  assert.equal((await stored()).map_id, null);
  assert.equal((await stored()).workspace.width_m, incomplete.workspace.width_m);
  assert.equal((await stored()).workspace.height_m, incomplete.workspace.height_m);
  assert.equal(await continueButton.isVisible(), true, 'The selected incomplete mission remains active after the old child poll resolves');
  await page.unroute(oldChildRoute);
  report.checks.delayed_old_poll = { held_child_id: child.id, selected_archive_id: fixtureID, config_and_map_preserved: true };
  const continuationStart = await clickPost(`/api/missions/${fixtureID}/continue`, continueButton);
  const continuation = await completed(continuationStart.body.id);
  assert.equal(continuation.stage.kind, 'arrival_continuation');
  assert.equal(continuation.stage.parent_id, fixtureID);
  continuity(incomplete, continuation, { origin_x_m: 0, origin_y_m: 0 });
  assert.ok(continuation.frames.at(-1).agents.some((agent, i) => Math.hypot(agent.x - continuation.frames[0].agents[i].x, agent.y - continuation.frames[0].agents[i].y) > .001));
  report.checks.continuation = { source_id: fixtureID, continued_id: continuation.id, positions_and_battery_preserved: true, actual_motion: true, source_budget_steps: incomplete.summary.actual_steps };
  report.checks.archives = { original_mission_exists: (await get(`/api/missions/${fixtureID}`)).id === fixtureID, original_map_exists: (await get(`/api/maps/${firstMap.id}`)).id === firstMap.id, saved_mission_count: (await get('/api/missions')).length };
  assert.deepEqual(jsErrors, []);
  assert.deepEqual(apiErrors, []);
  assert.deepEqual(transportErrors, []);
  report.passed = true;
} catch (error) {
  report.error = error.stack || String(error);
  if (page) await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {});
  process.exitCode = 1;
} finally {
  releaseDelayedPolls();
  report.browser_errors = jsErrors; report.api_errors = apiErrors; report.transport_errors = transportErrors;
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (browser) await browser.close();
  if (server && server.exitCode === null) {
    const stopped = new Promise(resolve => server.once('exit', resolve));
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(server.pid), '/T', '/F'], { windowsHide: true });
    else server.kill('SIGTERM');
    await stopped;
  }
  fs.closeSync(log);
  // Retain the isolated server log/data for diagnosis. No user files are removed.
}
