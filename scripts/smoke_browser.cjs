/* Functional browser smoke against the real running AGMEN application. */
const { createRequire } = require('node:module');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
let playwright;
try { playwright = require('../frontend/node_modules/playwright'); }
catch (error) {
  if (!process.env.AGMEN_PLAYWRIGHT_MODULE) throw error;
  playwright = createRequire(path.join(process.env.AGMEN_PLAYWRIGHT_MODULE, 'package.json'))('playwright');
}
const base = process.env.AGMEN_URL || 'http://127.0.0.1:7860';
const output = path.resolve(__dirname, '../test-results');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const options = { headless: true };
  if (process.env.AGMEN_CHROME) options.executablePath = process.env.AGMEN_CHROME;
  else if (process.platform === 'win32') options.channel = 'chrome';
  const browser = await playwright.chromium.launch(options);
  const context = await browser.newContext({ viewport: { width: 1512, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  page.on('response', response => {
    if (response.url().includes('/api/') && response.status() >= 400) failures.push(`${response.status()} ${response.url()}`);
  });
  try {
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.getByRole('heading', { name: 'Центр управления миссией' }).waitFor();
    const response = page.waitForResponse(r => r.url().endsWith('/api/missions') && r.request().method() === 'POST');
    await page.getByRole('button', { name: /Рассчитать миссию/ }).click();
    const created = await (await response).json();
    assert.ok(created.id, 'Mission was created by the actual server');
    await page.waitForFunction(async id => {
      const r = await fetch(`/api/missions/${id}`);
      return (await r.json()).status === 'completed';
    }, created.id, { timeout: 60000 });
    await page.locator('.fleet-table tbody tr').first().waitFor({ timeout: 15000 });
    await page.screenshot({ path: path.join(output, 'mission-desktop.png'), fullPage: true });
    const loaded = await context.request.get(`${base}/api/missions/${created.id}`);
    const mission = await loaded.json();
    assert.ok(mission.frames.length > 1);
    assert.equal(mission.frames[0].agents.length, mission.config.drone_count);
    await page.getByRole('button', { name: 'Аналитика', exact: true }).click();
    await page.getByRole('heading', { name: 'Аналитика миссий' }).waitFor();
    assert.ok(await page.locator('svg[role="img"]').count() > 0, 'Analytics contains measured charts');
    await page.screenshot({ path: path.join(output, 'analytics-desktop.png'), fullPage: true });
    await page.getByRole('button', { name: 'AI лаборатория', exact: true }).click();
    await page.getByRole('heading', { name: 'AI лаборатория', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'lab-desktop.png'), fullPage: true });
    await page.getByRole('button', { name: 'AGMEN EDGE', exact: true }).click();
    await page.getByRole('heading', { name: 'AGMEN EDGE CONTROL' }).waitFor();
    const packageResponse = page.waitForResponse(r => r.url().endsWith('/api/edge/packages') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Подготовить автономную миссию', exact: true }).click();
    const edgePackage = await (await packageResponse).json();
    assert.ok(edgePackage.id, 'A real mission package was built');
    const deviceResponse = page.waitForResponse(r => r.url().endsWith('/api/edge/devices') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Загрузить', exact: true }).first().click();
    const device = await (await deviceResponse).json();
    async function readDevice() {
      return (await context.request.get(`${base}/api/edge/devices/${device.id}`)).json();
    }
    async function runDevice(minimumStep) {
      const runResponse = page.waitForResponse(r => r.url().endsWith(`/api/edge/devices/${device.id}/run`) && r.request().method() === 'POST');
      await page.getByRole('button', { name: 'Выполнить 20 шагов', exact: true }).click();
      assert.equal((await runResponse).status(), 202);
      await page.waitForFunction(async ({ id, step }) => {
        const value = await (await fetch(`/api/edge/devices/${id}`)).json();
        return value.job_status === 'idle' && value.step >= step;
      }, { id: device.id, step: minimumStep }, { timeout: 30000 });
      return readDevice();
    }
    await runDevice(20);
    await page.getByRole('button', { name: 'Потеря интернета', exact: true }).click();
    await page.waitForFunction(async id => !(await (await fetch(`/api/edge/devices/${id}`)).json()).connectivity.internet, device.id);
    const offlineStart = await readDevice();
    const offline = await runDevice(40);
    assert.equal(offline.connectivity.internet, false);
    assert.notDeepEqual(offline.position_m, offlineStart.position_m, 'Local execution continues without Internet');
    await page.getByRole('button', { name: 'Потеря связи', exact: true }).click();
    await page.waitForFunction(async id => !(await (await fetch(`/api/edge/devices/${id}`)).json()).connectivity.ground_radio, device.id);
    const staleStep = (await readDevice()).telemetry.step;
    const lost = await runDevice(60);
    assert.equal(lost.mode, 'LINK LOST');
    assert.equal(lost.telemetry_kind, 'last_known');
    assert.equal(lost.telemetry.step, staleStep, 'Unconfirmed position does not replace received telemetry');
    await page.screenshot({ path: path.join(output, 'edge-link-lost.png'), fullPage: true });
    await page.locator('.connectivity-grid button').filter({ hasText: 'Наземный радиоканал' }).click();
    await page.getByRole('button', { name: 'Синхронизировать', exact: true }).click();
    const synced = await readDevice();
    assert.ok(synced.ground_receipt, 'A separate actual ground journal receipt exists');
    assert.equal((await context.request.get(`${base}/api/edge/devices/${device.id}/journal`)).status(), 200);
    await page.screenshot({ path: path.join(output, 'edge-desktop.png'), fullPage: true });
    await page.getByRole('button', { name: 'Миссии', exact: true }).click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, 'mission-mobile.png'), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
    assert.equal(overflow, false, 'Mobile page fits the viewport');
    assert.deepEqual(failures, [], 'No browser runtime or API errors');
    fs.writeFileSync(path.join(output, 'browser-smoke.json'), JSON.stringify({ passed: true, base, mission_id: created.id, frames: mission.frames.length, drones: mission.config.drone_count, edge_device: device.id, edge_package: edgePackage.id, offline_steps: offline.step, radio_loss_last_confirmed_step: staleStep, synced_entries: synced.synced_entries, browser_errors: failures, screenshots: ['mission-desktop.png', 'analytics-desktop.png', 'lab-desktop.png', 'edge-desktop.png', 'edge-link-lost.png', 'mission-mobile.png'] }, null, 2));
    console.log(`Browser smoke PASS: mission ${created.id}, ${mission.frames.length} real frames, desktop/mobile and four sections.`);
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
    fs.writeFileSync(path.join(output, 'browser-smoke-failure.txt'), `${error.stack}\n${JSON.stringify(failures)}\n${await page.locator('body').innerText()}`);
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
