/** Real UI regression for uploaded territories and confirmed no-go geometry. */
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = path.join(root, 'test-results', 'territory-qa');
fs.mkdirSync(out, { recursive: true });
const base = process.env.AGMEN_URL || 'http://127.0.0.1:7860';
const missionName = `Territory browser regression ${Date.now()}`;
const localPython = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.AGMEN_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const fixture = path.join(out, 'territory-fixture.png');
const fixtureCode = `from PIL import Image, ImageDraw
import sys
im=Image.new('RGB',(1200,900),(237,232,211))
d=ImageDraw.Draw(im)
for x in range(0,1200,100): d.line((x,0,x,900),fill=(210,205,186),width=2)
for y in range(0,900,100): d.line((0,y,1200,y),fill=(210,205,186),width=2)
d.rectangle((530,120,620,205),fill=(48,46,35))
d.rectangle((850,700,980,790),fill=(80,160,200))
im.save(sys.argv[1])`;
const fixtureResult = spawnSync(python, ['-c', fixtureCode, fixture], { cwd: root, windowsHide: true, encoding: 'utf8' });
assert.equal(fixtureResult.status, 0, fixtureResult.stderr);

const browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? { channel: 'chrome' } : {}) });
const page = await browser.newPage({ viewport: { width: 1512, height: 1100 } });
const jsErrors = [], apiErrors = [];
page.on('pageerror', error => jsErrors.push(error.message));
page.on('response', response => {
  if (new URL(response.url()).pathname.startsWith('/api/') && response.status() >= 400) {
    apiErrors.push(`${response.status()} ${response.request().method()} ${response.url()}`);
  }
});
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, description, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await check()) return; await delay(100); }
  throw new Error(`Timed out: ${description}`);
}
function waitResponse(route) {
  return page.waitForResponse(response => new URL(response.url()).pathname === route && response.request().method() === 'POST');
}
async function postWithButton(route, button) {
  const waiting = waitResponse(route);
  await button.click();
  const response = await waiting;
  const data = await response.json();
  assert.ok(response.ok(), `${route}: ${response.status()} ${JSON.stringify(data)}`);
  return data;
}
async function shapeCount(count) {
  await until(async () => await page.locator('.terrain-restricted').count() === count, `${count} confirmed restrictions`);
  assert.match(await page.locator('.territory-actions .shape-tag').innerText(), new RegExp(`^${count} / 32`));
}
async function mapPoint(x, y) {
  const svg = page.getByLabel('Интерактивное поле симуляции', { exact: true });
  await svg.scrollIntoViewIfNeeded();
  return svg.evaluate((element, point) => {
    const matrix = element.getScreenCTM();
    const translated = new DOMPoint(44 + point.x * 880, 28 + point.y * 578).matrixTransform(matrix);
    return { x: translated.x, y: translated.y };
  }, { x, y });
}
async function dragShape(tool, from, to) {
  await page.getByRole('button', { name: `Запретная зона: ${tool}`, exact: true }).click();
  const start = await mapPoint(...from), end = await mapPoint(...to);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 12 });
  await page.mouse.up();
}
async function polygon() {
  await page.getByRole('button', { name: 'Запретная зона: Многоугольник', exact: true }).click();
  for (const point of [[.38, .55], [.42, .60], [.37, .65]]) {
    const screen = await mapPoint(...point);
    await page.mouse.click(screen.x, screen.y);
  }
  await page.getByRole('button', { name: 'Завершить контур', exact: true }).click();
}
async function missionCompleted(identifier) {
  let result;
  await until(async () => {
    const response = await page.request.get(`${base}/api/missions/${identifier}`);
    assert.ok(response.ok());
    result = await response.json();
    assert.notEqual(result.status, 'failed', result.error);
    return result.status === 'completed';
  }, `mission ${identifier} completion`, 60000);
  return result;
}
async function captureTheme(theme, label) {
  await page.setViewportSize({ width: 1512, height: 1100 });
  await page.getByRole('button', { name: 'Настройки', exact: true }).click();
  await page.getByRole('button', { name: label, exact: true }).click();
  assert.equal(await page.locator('html').getAttribute('data-theme'), theme);
  assert.equal(await page.locator('html').evaluate(element => getComputedStyle(element).colorScheme), theme);
  assert.equal(await page.evaluate(() => localStorage.getItem('agmen-theme')), theme);
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  assert.equal(await page.locator('html').getAttribute('data-theme'), theme, 'Theme persists after a real reload');
  await page.getByRole('button', { name: 'Миссии', exact: true }).click();
  await page.getByRole('button', { name: 'Запретная зона: Выбор', exact: true }).click();
  const pause = page.getByRole('button', { name: 'Пауза', exact: true });
  if (await pause.count()) await pause.click();
  const toast = page.getByRole('button', { name: 'Закрыть уведомление', exact: true });
  if (await toast.count()) await toast.click();
  await page.screenshot({ path: path.join(out, `territory-${theme}-desktop.png`), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await until(async () => await page.locator('.mission-map').evaluate(element => element.getBoundingClientRect().width > 0), 'mobile map layout');
  const sizes = await page.evaluate(() => ({ viewport: innerWidth, scroll: document.documentElement.scrollWidth }));
  assert.ok(sizes.scroll <= sizes.viewport + 1, `${theme} mobile document overflows: ${JSON.stringify(sizes)}`);
  const map = await page.locator('.mission-map').evaluate(element => {
    const bounds = element.getBoundingClientRect(), parent = element.parentElement.getBoundingClientRect();
    return { left: bounds.left, right: bounds.right, parentLeft: parent.left, parentRight: parent.right };
  });
  assert.ok(map.left >= map.parentLeft - 1 && map.right <= map.parentRight + 1, 'Entire territory field fits its mobile container');
  await page.screenshot({ path: path.join(out, `territory-${theme}-mobile.png`), fullPage: true });
  return { theme, sizes, map };
}
try {
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await page.getByRole('button', { name: 'Миссии', exact: true }).click();
  await page.locator('.territory-heading > button').first().click();
  await page.getByLabel('Ширина территории, м', { exact: true }).fill('120');
  await page.getByLabel('Высота территории, м', { exact: true }).fill('90');
  const uploadedResponse = waitResponse('/api/maps');
  await page.getByLabel('Файл карты территории', { exact: true }).setInputFiles(fixture);
  const uploadResponse = await uploadedResponse;
  assert.equal(uploadResponse.status(), 201, await uploadResponse.text());
  const territory = await uploadResponse.json();
  assert.equal(territory.width_m, 120);
  assert.equal(territory.height_m, 90);
  assert.ok(territory.suggestions.length > 0, 'Fixture dark object generates a real analysis candidate');
  assert.equal(territory.analysis.requires_confirmation, true);
  await page.locator('.map-background').waitFor();
  assert.match(await page.locator('.map-background').getAttribute('href'), /^blob:/);
  await shapeCount(0);
  assert.equal(await page.locator('.terrain-suggestion').count(), territory.suggestions.length);

  // Changing the analysis method produces real backend results, never confirmed zones.
  await page.getByLabel('Метод анализа карты', { exact: true }).selectOption('water');
  const water = await postWithButton(`/api/maps/${territory.id}/analyze`, page.getByRole('button', { name: 'Найти препятствия', exact: true }));
  assert.ok(water.suggestions.some(region => region.x > .65 && region.y > .65), 'Water heuristic finds the blue fixture region in image coordinates');
  await shapeCount(0);
  await page.getByLabel('Метод анализа карты', { exact: true }).selectOption('dark');
  await postWithButton(`/api/maps/${territory.id}/analyze`, page.getByRole('button', { name: 'Найти препятствия', exact: true }));
  await page.getByRole('button', { name: 'Добавить в запреты', exact: true }).first().click();
  await shapeCount(1);
  await dragShape('Прямоугольник', [.30, .25], [.34, .35]);
  await shapeCount(2);
  await dragShape('Круг', [.58, .29], [.62, .29]);
  await shapeCount(3);
  await polygon();
  await shapeCount(4);
  await page.getByRole('button', { name: 'Удалить последнюю запретную зону', exact: true }).click();
  await shapeCount(3);
  await polygon();
  await shapeCount(4);
  await page.getByRole('button', { name: 'Удалить запрет 2', exact: true }).click();
  await shapeCount(3);
  await dragShape('Прямоугольник', [.30, .25], [.34, .35]);
  await shapeCount(4);

  // Editing map dimensions is a draft until Apply succeeds; geometry must not
  // silently change scale while a number field is being edited.
  const circle = page.locator('.terrain-restricted ellipse');
  const originalRadiusX = Number(await circle.getAttribute('rx'));
  await page.getByLabel('Ширина территории, м', { exact: true }).fill('9');
  assert.equal(await page.getByRole('button', { name: 'Применить масштаб', exact: true }).isDisabled(), true);
  assert.equal(Number(await circle.getAttribute('rx')), originalRadiusX, 'Invalid draft dimensions never distort existing geometry');
  await page.getByLabel('Ширина территории, м', { exact: true }).fill('160');
  assert.equal(Number(await circle.getAttribute('rx')), originalRadiusX);
  assert.match(await page.locator('.map-panel > .panel-header .subtle').innerText(), /120 × 90/);
  const calibrated = await postWithButton(`/api/maps/${territory.id}/calibrate`, page.getByRole('button', { name: 'Применить масштаб', exact: true }));
  assert.equal(calibrated.width_m, 160);
  await until(async () => Math.abs(Number(await circle.getAttribute('rx')) - originalRadiusX * .75) < .001, 'circle uses successfully applied physical scale');
  await page.getByLabel('Ширина территории, м', { exact: true }).fill('120');
  await postWithButton(`/api/maps/${territory.id}/calibrate`, page.getByRole('button', { name: 'Применить масштаб', exact: true }));
  await until(async () => Math.abs(Number(await circle.getAttribute('rx')) - originalRadiusX) < .001, 'original physical scale restored');

  const storedDraft = await page.evaluate(() => JSON.parse(localStorage.getItem('agmen-territory')));
  assert.equal(storedDraft.map_id, territory.id);
  assert.equal(storedDraft.restricted_zones.length, 4);
  assert.deepEqual(storedDraft.restricted_zones.map(zone => zone.kind).sort(), ['circle', 'polygon', 'rectangle', 'rectangle']);
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await page.locator('.map-background').waitFor();
  await shapeCount(4);
  await page.getByRole('button', { name: 'Pro', exact: true }).click();
  await page.getByRole('button', { name: 'Math', exact: true }).click();
  await page.getByLabel('Название миссии', { exact: true }).fill(missionName);
  await page.getByLabel('Сценарий', { exact: true }).selectOption('distribution');
  await page.getByLabel('Препятствия', { exact: true }).selectOption('none');
  await page.getByLabel('Длительность расчёта', { exact: true }).selectOption('20');
  const count = page.getByLabel('Количество аппаратов', { exact: true });
  await count.focus();
  await count.press('Home');
  const missionResponse = waitResponse('/api/missions');
  await page.getByRole('button', { name: 'Рассчитать миссию', exact: true }).click();
  const createdResponse = await missionResponse;
  const submitted = createdResponse.request().postDataJSON();
  assert.equal(createdResponse.status(), 202, await createdResponse.text());
  assert.equal(submitted.map_id, territory.id);
  assert.equal(submitted.restricted_zones.length, 4);
  assert.equal(submitted.drone_count, 5);
  assert.ok(!('map_snapshot' in submitted), 'Server-only territory snapshot is not resubmitted');
  const created = await createdResponse.json();
  const record = await missionCompleted(created.id);
  assert.equal(record.workspace.width_m, 120);
  assert.equal(record.workspace.height_m, 90);
  assert.ok(record.obstacle.flat().some(Boolean));
  assert.equal(record.territory.id, territory.id);
  assert.equal(record.restriction_policy.swept_segment_checks, true);
  for (const frame of record.frames) {
    for (const agent of frame.agents) {
      assert.ok(!record.obstacle_rects.some(rect => agent.x >= rect.x && agent.x <= rect.x + rect.width && agent.y >= rect.y && agent.y <= rect.y + rect.height), `Vehicle ${agent.id} at step ${frame.step} stays outside confirmed restrictions`);
    }
  }
  await until(async () => await page.locator('.drone-marker').count() === 5, 'recorded mission rendered');
  // A source map may be recalibrated later. Opening the archived mission must
  // retain that mission's original 120 x 90 m snapshot, including after reload.
  const recalibration = await page.request.post(`${base}/api/maps/${territory.id}/calibrate`, { data: { width_m: 160, height_m: 90 } });
  assert.ok(recalibration.ok());
  await page.getByRole('button', { name: 'Аналитика', exact: true }).click();
  const restored = page.waitForResponse(response => new URL(response.url()).pathname === `/api/missions/${created.id}`);
  await page.locator('tr').filter({ hasText: missionName }).getByRole('button', { name: 'Открыть', exact: true }).click();
  assert.ok((await restored).ok());
  await page.getByRole('button', { name: 'Миссии', exact: true }).click();
  await until(async () => /120 × 90/.test(await page.locator('.map-panel > .panel-header .subtle').innerText()), 'archived mission keeps its original dimensions');
  await page.reload({ waitUntil: 'networkidle' });
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor();
  await shapeCount(4);
  assert.match(await page.locator('.map-panel > .panel-header .subtle').innerText(), /120 × 90/);
  // An actual second calculation verifies saved configs with a server snapshot.
  const repeated = await postWithButton('/api/missions', page.getByRole('button', { name: 'Рассчитать миссию', exact: true }));
  const repeatedRecord = await missionCompleted(repeated.id);
  assert.equal(repeatedRecord.territory.id, territory.id);
  assert.equal(repeatedRecord.workspace.width_m, 160);
  await until(async () => /160 × 90/.test(await page.locator('.map-panel > .panel-header .subtle').innerText()), 'new mission displays the server-resolved current map dimensions');

  const themeLayouts = [];
  themeLayouts.push(await captureTheme('light', 'Светлая'));
  themeLayouts.push(await captureTheme('dark', 'Тёмная'));
  await page.setViewportSize({ width: 1512, height: 1100 });
  await page.getByRole('button', { name: 'AGMEN EDGE', exact: true }).click();
  const metadata = await postWithButton('/api/edge/packages', page.getByRole('button', { name: 'Подготовить автономную миссию', exact: true }));
  const archive = await page.request.get(`${base}/api/edge/packages/${metadata.id}/download`);
  assert.ok(archive.ok());
  const archivePath = path.join(out, 'territory-edge.zip');
  fs.writeFileSync(archivePath, await archive.body());
  const inspect = spawnSync(python, ['-c', "import json,sys;from edge import load_package;p=load_package(sys.argv[1]);print(json.dumps({'map_id':p['map']['map_id'],'workspace':p['map']['workspace'],'zones':len(p['map']['restricted_zones']),'obstacle_cells':sum(sum(row) for row in p['map']['obstacle'])}))", archivePath], { cwd: root, windowsHide: true, encoding: 'utf8' });
  assert.equal(inspect.status, 0, inspect.stderr);
  const offline = JSON.parse(inspect.stdout.trim());
  assert.equal(offline.map_id, territory.id);
  assert.equal(offline.workspace.width_m, 160);
  assert.equal(offline.workspace.height_m, 90);
  assert.equal(offline.zones, 4);
  assert.ok(offline.obstacle_cells > 0);
  assert.deepEqual(jsErrors, []);
  assert.deepEqual(apiErrors, []);
  const report = { passed: true, territory_id: territory.id, mission_id: created.id, repeated_mission_id: repeated.id,
    confirmed_zones: submitted.restricted_zones, frames_checked: record.frames.length, candidate_confirmation_required: true,
    calibration_draft_verified: true, archived_workspace: record.workspace, repeated_workspace: repeatedRecord.workspace,
    offline_package: offline, theme_layouts: themeLayouts, browser_errors: jsErrors, api_errors: apiErrors };
  fs.writeFileSync(path.join(out, 'territory-browser-qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(out, 'territory-qa-failure.png'), fullPage: true });
  console.error((await page.locator('body').innerText()).slice(-7000));
  console.error({ jsErrors, apiErrors });
  throw error;
} finally {
  await browser.close();
}
