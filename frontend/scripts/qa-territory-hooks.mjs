// Deterministic request-race checks against the shipped hook implementation.
// A small hook lifecycle runner substitutes React scheduling and deferred API I/O;
// real-browser layout/input coverage lives in qa-arrival-stage.mjs.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let active;
const storage = new Map(), records = new Map(), gates = new Map();
const window = new EventTarget();
const react = {
  useState(initial) {
    const runner = active, index = runner.index++;
    if (!runner.hooks[index]) runner.hooks[index] = { value: typeof initial === 'function' ? initial() : initial };
    return [runner.hooks[index].value, next => {
      const value = typeof next === 'function' ? next(runner.hooks[index].value) : next;
      if (!Object.is(value, runner.hooks[index].value)) { runner.hooks[index].value = value; runner.dirty = true; }
    }];
  },
  useRef(value) {
    const runner = active, index = runner.index++;
    if (!runner.hooks[index]) runner.hooks[index] = { current: value };
    return runner.hooks[index];
  },
  useEffect(effect, dependencies) {
    const runner = active, index = runner.index++, previous = runner.hooks[index];
    if (!previous || dependencies.some((value, i) => !Object.is(value, previous.dependencies[i]))) {
      runner.hooks[index] = { dependencies };
      runner.effects.push(() => { previous?.cleanup?.(); runner.hooks[index].cleanup = effect(); });
    }
  },
};
const api = async (route, data) => {
  const key = `${data === undefined ? 'GET' : 'POST'}:${route}`;
  if (gates.has(key)) return gates.get(key).promise;
  if (route === '/api/maps' && data === undefined) return [...records.values()];
  if (route.startsWith('/api/maps/')) return records.get(route.split('/')[3]);
  throw new Error(`Unexpected fixture request: ${key}`);
};
const exports = {};
const context = {
  exports,
  require(name) {
    if (name === 'react') return react;
    if (name === 'react/jsx-runtime') return { jsx() {}, jsxs() {} };
    if (name === 'lucide-react') return new Proxy({}, { get() { return function Icon() {}; } });
    if (name === './api') return { api, requestBlob: async () => new Blob(['fixture PNG']) };
    throw new Error(name);
  },
  localStorage: { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
  window, Event,
  URL: { createObjectURL: () => 'blob:fixture', revokeObjectURL() {} },
  crypto: { randomUUID: () => 'fixture-id-123' },
  FileReader: class { readAsDataURL() { this.result = 'data:image/png;base64,eA=='; queueMicrotask(() => this.onload()); } },
  console,
};
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(frontend, 'src/TerritoryMap.tsx'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
}).outputText, context);
class Runner {
  constructor(storageKey, online = true) {
    this.hooks = []; this.index = 0; this.effects = []; this.dirty = true;
    this.storageKey = storageKey; this.online = online; this.edits = []; this.errors = [];
  }
  render() {
    active = this; this.index = 0;
    this.editor = exports.useTerritory(error => this.errors.push(error), this.online, reason => this.edits.push(reason), this.storageKey);
    active = undefined; this.dirty = false;
    for (const effect of this.effects.splice(0)) effect();
  }
  async flush() {
    for (let i = 0; i < 30; i++) { if (this.dirty) this.render(); await Promise.resolve(); }
    return this.editor;
  }
}
function gate(key) {
  let resolve;
  const promise = new Promise(ready => { resolve = ready; });
  const deferred = { promise, resolve(value) { gates.delete(key); resolve(value); } };
  gates.set(key, deferred); return deferred;
}

const map = { id: 'map1', name: 'fixture.png', width_m: 120, height_m: 90, image_url: '/fixture-image', suggestions: [{ id: 's1', kind: 'rectangle', x: .1, y: .1, width: .1, height: .1 }] };
records.set(map.id, map);
const shape = { id: 'zone1', kind: 'circle', x: .5, y: .5, radius: .1 };
const runner = new Runner('qa-origin');
let editor = await runner.flush();
editor.restore('map1', [shape], { width_m: 80, height_m: 60 });
editor = await runner.flush();
assert.equal(editor.width, 80); assert.equal(editor.height, 60); assert.equal(editor.shapes.length, 1);
editor.setWidth(200); editor = await runner.flush();
assert.equal(editor.width, 80); assert.equal(editor.draftWidth, 200);
const reloaded = new Runner('qa-origin');
const snapshot = await reloaded.flush();
assert.equal(snapshot.width, 80); assert.equal(snapshot.shapes.length, 1);
const destination = new Runner('qa-destination');
let second = await destination.flush();
assert.equal(second.mapId, null); assert.equal(second.width, 80); assert.equal(second.shapes.length, 0);

const analysis = gate('POST:/api/maps/map1/analyze');
const pendingAnalysis = editor.analyze();
await runner.flush(); runner.editor.clearZones(); await runner.flush();
analysis.resolve(map); await pendingAnalysis; editor = await runner.flush();
assert.equal(editor.shapes.length, 0); assert.equal(editor.map.suggestions.length, 0);
assert.equal(editor.width, 80); assert.equal(editor.mapId, 'map1'); assert.equal(editor.image, 'blob:fixture');
assert.equal(runner.edits.at(-1), 'shape');
editor.restore('map1', [], { width_m: 80, height_m: 60 }); editor = await runner.flush();
assert.equal(editor.map.suggestions.length, 0, 'Restoring a result on the same map preserves the cleared-proposal intent');
assert.equal(JSON.parse(storage.get('qa-origin')).suggestions_cleared, true);
const clearedReload = new Runner('qa-origin');
assert.equal((await clearedReload.flush()).map.suggestions.length, 0);

const upload = gate('POST:/api/maps');
const pendingUpload = editor.upload({ target: { files: [{ type: 'image/png', size: 50, name: 'new.png' }], value: 'file' } });
await runner.flush(); runner.editor.resetMap(); editor = await runner.flush();
assert.equal(editor.mapId, null); assert.equal(editor.width, 80); assert.equal(editor.height, 60);
assert.equal(editor.method, 'dark'); assert.equal(editor.sensitivity, .5); assert.equal(editor.expanded, true); assert.equal(editor.busy, false);
upload.resolve({ ...map, id: 'new-map' }); await pendingUpload; editor = await runner.flush();
assert.equal(editor.mapId, null); assert.equal(editor.image, ''); assert.equal(editor.map, null); assert.equal(editor.shapes.length, 0);
assert.equal(runner.edits.at(-1), 'reset');

const metadata = gate('GET:/api/maps/map1');
editor.restore('map1', [shape], { width_m: 120, height_m: 90 }); await runner.flush();
runner.editor.resetMap(); await runner.flush(); metadata.resolve(map); editor = await runner.flush();
assert.equal(editor.mapId, null); assert.equal(editor.map, null); assert.equal(editor.image, ''); assert.equal(editor.width, 80);

editor.restore('map1', [shape], { width_m: 80, height_m: 60 }); editor = await runner.flush();
editor.setWidth(160); editor = await runner.flush();
const calibration = gate('POST:/api/maps/map1/calibrate');
const pendingCalibration = editor.calibrate(); await runner.flush();
runner.editor.resetMap(); await runner.flush(); calibration.resolve({ ...map, width_m: 160 });
await pendingCalibration; editor = await runner.flush();
assert.equal(editor.width, 80); assert.equal(editor.mapId, null); assert.equal(editor.map, null);

records.set('map2', { ...map, id: 'map2' });
await editor.refreshMaps(); await runner.flush();
window.dispatchEvent(new Event('agmen-maps-changed')); second = await destination.flush();
assert.ok(second.maps.some(value => value.id === 'map2'), 'A second editor refreshes its external catalog');
assert.equal(second.mapId, null, 'Catalog refresh does not activate a destination map');
assert.equal(JSON.parse(storage.get('qa-destination')).map_id, null);
assert.equal(records.get('map1'), map, 'Reset never changes archived server records');
assert.deepEqual(runner.errors, []); assert.deepEqual(destination.errors, []);
console.log(JSON.stringify({ passed: true, checks: ['mission_snapshot_scale', 'draft_scale_isolation', 'independent_storage_keys', 'clear_zones_analysis_race', 'clear_suggestions_same_map_restore', 'clear_suggestions_reload', 'reset_upload_race', 'reset_metadata_race', 'reset_calibration_race', 'external_map_catalog_refresh', 'archive_preservation'] }, null, 2));
