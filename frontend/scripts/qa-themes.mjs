// Live UI contrast/overflow checks. Uses an isolated browser context; no app data writes.
// Text is hidden only during the background capture, then restored immediately.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const base = process.env.AGMEN_URL || 'http://127.0.0.1:7860';
const out = path.join(root, 'test-results/theme-qa');
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? { channel: 'chrome' } : {}) });
const context = await browser.newContext({ viewport: { width: 1512, height: 1000 }, deviceScaleFactor: 1 });
await context.addInitScript(() => {
  if (localStorage.getItem('agmen-theme') === null) localStorage.setItem('agmen-theme', 'light');
  if (localStorage.getItem('agmen-pro') === null) localStorage.setItem('agmen-pro', 'true');
});
const page = await context.newPage();
const pageErrors = [];
page.on('pageerror', error => pageErrors.push(error.message));
const views = [
  ['mission', 'Миссии'], ['lab', 'AI лаборатория'], ['edge', 'AGMEN EDGE'],
  ['analytics', 'Аналитика'], ['settings', 'Настройки'],
];
const report = { base, checked_at: new Date().toISOString(), thresholds: { text: 4.5, functional_icons: 3 }, views: [], failures: [], browser_errors: pageErrors };

async function audit(theme, viewport, view) {
  const screenshotPath = path.join(out, `${theme}-${viewport}-${view}.png`);
  await page.screenshot({ path: screenshotPath, fullPage: true, animations: 'disabled' });
  // Gather normal computed colors before temporarily masking text and UI icons.
  const nodes = await page.evaluate(() => {
    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = colorCanvas.height = 1;
    const colorContext = colorCanvas.getContext('2d', { willReadFrequently: true });
    const rgba = css => {
      colorContext.clearRect(0, 0, 1, 1);
      colorContext.fillStyle = css;
      colorContext.fillRect(0, 0, 1, 1);
      const data = colorContext.getImageData(0, 0, 1, 1).data;
      return [data[0], data[1], data[2], data[3] / 255];
    };
    const selector = node => {
      const result = [];
      for (let e = node; e && result.length < 4; e = e.parentElement) {
        let part = e.tagName.toLowerCase();
        if (e.id) { result.unshift(`${part}#${CSS.escape(e.id)}`); break; }
        const classes = [...e.classList].slice(0, 3);
        if (classes.length) part += classes.map(c => `.${CSS.escape(c)}`).join('');
        else if (e.parentElement) part += `:nth-of-type(${[...e.parentElement.children].filter(n => n.tagName === e.tagName).indexOf(e) + 1})`;
        result.unshift(part);
      }
      return result.join(' > ');
    };
    const visible = e => {
      if (e.closest('[disabled],[aria-disabled="true"],script,style,option')) return false;
      let clipped = e.getBoundingClientRect();
      if (clipped.width <= 2 || clipped.height <= 2) return false;
      for (let parent = e; parent; parent = parent.parentElement) {
        const s = getComputedStyle(parent);
        if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0) return false;
        if (parent !== e && parent !== document.body && parent !== document.documentElement && /(hidden|clip|auto|scroll)/.test(s.overflowX)) {
          const r = parent.getBoundingClientRect();
          if (clipped.right <= r.left || clipped.left >= r.right) return false;
        }
        if (parent !== e && parent !== document.body && parent !== document.documentElement && /(hidden|clip|auto|scroll)/.test(s.overflowY)) {
          const r = parent.getBoundingClientRect();
          if (clipped.bottom <= r.top || clipped.top >= r.bottom) return false;
        }
      }
      return true;
    };
    const opacity = e => {
      let result = 1;
      for (let parent = e; parent; parent = parent.parentElement) result *= Number(getComputedStyle(parent).opacity);
      return result;
    };
    const rect = (r, element) => {
      let left = r.left, right = r.right, top = r.top, bottom = r.bottom;
      for (let parent = element?.parentElement; parent && parent !== document.body && parent !== document.documentElement; parent = parent.parentElement) {
        const s = getComputedStyle(parent), p = parent.getBoundingClientRect();
        if (/(hidden|clip|auto|scroll)/.test(s.overflowX)) { left = Math.max(left, p.left); right = Math.min(right, p.right); }
        if (/(hidden|clip|auto|scroll)/.test(s.overflowY)) { top = Math.max(top, p.top); bottom = Math.min(bottom, p.bottom); }
      }
      return { x: left + scrollX, y: top + scrollY, width: right - left, height: bottom - top };
    };
    const exposedFractions = (box, e) => [.25, .5, .75].filter(fraction => {
      const x = box.x - scrollX + box.width * fraction, y = box.y - scrollY + box.height / 2;
      // Full-page capture also includes content below the initial viewport.
      if (x < 0 || x >= innerWidth || y < 0 || y >= innerHeight) return true;
      const top = document.elementFromPoint(x, y);
      return !top || e.contains(top) || top.contains(e) || (e.closest('svg') && e.closest('svg') === top.closest('svg'));
    });
    const found = [];
    const tree = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (tree.nextNode()) {
      const node = tree.currentNode;
      const e = node.parentElement;
      const label = node.textContent.replace(/\s+/g, ' ').trim();
      if (!label || !e || !visible(e)) continue;
      const symbol = /^[↗↖↙↘↑↓←→−+]$/.test(label);
      if (symbol && e.closest('button')?.textContent.replace(label, '').trim()) continue; // An arrow next to a complete action label is decorative.
      const range = document.createRange();
      range.selectNodeContents(node);
      const boxes = [...range.getClientRects()].map(r => rect(r, e))
        .filter(r => r.width > 2 && r.height > 2)
        .map(box => ({ ...box, fractions: exposedFractions(box, e) }))
        .filter(box => box.fractions.length);
      if (!boxes.length) continue;
      const style = getComputedStyle(e);
      const color = e.closest('svg') ? style.fill : style.color;
      if (color === 'none') continue;
      // An SVG text outline is the immediate backing for the ink (map labels),
      // even when a sample point falls between glyphs over a terrain image.
      const halo = e.closest('svg') && style.stroke !== 'none' && Number.parseFloat(style.strokeWidth) >= 1 && style.paintOrder.startsWith('stroke')
        ? rgba(style.stroke).map((v, i) => i === 3 ? v * Number(style.strokeOpacity) : v)
        : null;
      found.push({ kind: symbol ? 'icon' : 'text', label: label.slice(0, 140), selector: selector(e), cssColor: color, rgba: rgba(color), halo, opacity: opacity(e), boxes, threshold: symbol ? 3 : 4.5 });
    }
    for (const e of document.querySelectorAll('input:not([type="range"]):not([type="checkbox"]):not([type="file"]),select,textarea')) {
      if (!visible(e)) continue;
      const placeholder = !e.value && e.placeholder;
      const style = getComputedStyle(e, placeholder ? '::placeholder' : null);
      const label = e.getAttribute('aria-label') || e.value || e.placeholder || e.tagName;
      const r = e.getBoundingClientRect();
      found.push({ kind: 'text', label, selector: selector(e), cssColor: style.color, rgba: rgba(style.color), opacity: opacity(e) * (placeholder ? Number(style.opacity) : 1), boxes: [{ x: r.left + scrollX + 8, y: r.top + scrollY + 4, width: Math.max(3, r.width - 20), height: Math.max(3, r.height - 8) }], threshold: 4.5 });
    }
    // Standalone icon controls convey an action visually and need non-text contrast.
    for (const svg of document.querySelectorAll('button > svg.lucide, [role="button"] > svg.lucide')) {
      const button = svg.parentElement;
      if (!visible(svg) || button.textContent.trim()) continue;
      const painted = [...svg.querySelectorAll('path,line,polyline,circle,rect,polygon')].find(e => getComputedStyle(e).stroke !== 'none');
      if (!painted) continue;
      const style = getComputedStyle(painted);
      found.push({ kind: 'icon', label: button.getAttribute('aria-label') || button.title || 'icon control', selector: selector(svg), cssColor: style.stroke, rgba: rgba(style.stroke), opacity: opacity(painted) * Number(style.strokeOpacity), boxes: [rect(svg.getBoundingClientRect(), svg)], threshold: 3 });
    }
    return found;
  });
  const mask = await page.addStyleTag({ content: '* { -webkit-text-fill-color: transparent !important; text-shadow: none !important; caret-color: transparent !important; } svg text { fill: transparent !important; } svg.lucide { visibility: hidden !important; }' });
  let background;
  try { background = await page.screenshot({ fullPage: true, animations: 'disabled' }); }
  finally { await mask.evaluate(e => e.remove()); }
  const measurements = await page.evaluate(async ({ encoded, nodes }) => {
    const picture = new Image();
    picture.src = `data:image/png;base64,${encoded}`;
    await picture.decode();
    const canvas = document.createElement('canvas');
    canvas.width = picture.naturalWidth; canvas.height = picture.naturalHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(picture, 0, 0);
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const luminance = rgb => rgb.map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4).reduce((s, v, i) => s + v * [.2126, .7152, .0722][i], 0);
    const contrast = (a, b) => { const la = luminance(a), lb = luminance(b); return (Math.max(la, lb) + .05) / (Math.min(la, lb) + .05); };
    const hex = rgb => '#' + rgb.map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
    return nodes.map(node => {
      const a = node.rgba[3] * node.opacity;
      const samples = node.boxes.flatMap(box => (box.fractions || [.25, .5, .75]).map(xFraction => {
        const x = Math.max(0, Math.min(canvas.width - 1, Math.round(box.x + box.width * xFraction)));
        const y = Math.max(0, Math.min(canvas.height - 1, Math.round(box.y + box.height * .5)));
        const index = (y * canvas.width + x) * 4;
        let bg = [pixels[index], pixels[index + 1], pixels[index + 2]];
        if (node.halo) {
          const haloAlpha = node.halo[3] * node.opacity;
          bg = node.halo.slice(0, 3).map((v, i) => v * haloAlpha + bg[i] * (1 - haloAlpha));
        }
        const fg = node.rgba.slice(0, 3).map((v, i) => v * a + bg[i] * (1 - a));
        return { ratio: contrast(fg, bg), foreground: hex(fg), background: hex(bg) };
      }));
      const worst = samples.reduce((a, b) => a.ratio < b.ratio ? a : b);
      return { kind: node.kind, label: node.label, selector: node.selector, cssColor: node.cssColor, opacity: node.opacity, threshold: node.threshold, ...worst, ratio: Number(worst.ratio.toFixed(3)), passes: worst.ratio >= node.threshold };
    });
  }, { encoded: background.toString('base64'), nodes });
  const overflow = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth }));
  const failures = measurements.filter(m => !m.passes).map(m => ({ theme, viewport, view, ...m }));
  if (overflow.document > overflow.viewport + 2) failures.push({ theme, viewport, view, kind: 'overflow', ...overflow });
  report.failures.push(...failures);
  report.views.push({ theme, viewport, view, checked_text: measurements.filter(m => m.kind === 'text').length, checked_icons: measurements.filter(m => m.kind === 'icon').length, failures: failures.length, overflow, screenshot: screenshotPath, measurements });
  console.log(`${theme}/${viewport}/${view}: ${measurements.length} checks, ${failures.length} failures`);
}

try {
  await page.goto(base);
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor({ timeout: 60000 });
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
  for (const theme of ['light', 'dark']) {
    if (await page.locator('html').getAttribute('data-theme') !== theme) {
      await page.getByRole('button', { name: 'Сменить тему', exact: true }).click();
      await page.waitForFunction(expected => document.documentElement.dataset.theme === expected, theme);
    }
    for (const [viewport, size] of [['desktop', { width: 1512, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
      await page.setViewportSize(size);
      for (const [view, label] of views) {
        await page.getByRole('button', { name: label, exact: true }).click();
        await page.evaluate(() => scrollTo(0, 0));
        await page.waitForTimeout(250);
        await audit(theme, viewport, view);
      }
    }
  }
  assert.equal(await page.evaluate(() => localStorage.getItem('agmen-theme')), 'dark');
  await page.reload();
  await page.locator('.engine-status').filter({ hasText: 'LOCAL ENGINE' }).waitFor({ state: 'attached', timeout: 60000 });
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark', 'Theme selection persists after reload');
  report.theme_persistence = true;
  report.passed = report.failures.length === 0 && pageErrors.length === 0;
} catch (error) {
  report.error = error.message;
  report.passed = false;
  await page.screenshot({ path: path.join(out, 'error.png'), fullPage: true }).catch(() => {});
} finally {
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: report.passed, views: report.views.length, failures: report.failures.length, browser_errors: report.browser_errors, error: report.error, report: path.join(out, 'report.json'), examples: report.failures.slice(0, 12) }, null, 2));
if (!report.passed) process.exitCode = 1;
