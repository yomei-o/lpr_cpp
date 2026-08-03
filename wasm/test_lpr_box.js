// End-to-end test of the WASM path that plate.html's manual box uses:
// full-frame RGBA + a box -> fn_recognize_box -> 9 argmax + 9 agreement scores.
//
// It exists because the browser page is the only place that path runs, and a page cannot be
// asserted on in CI. The native lpr_infer reads the same photo through the same header
// (pure/lpr_tta.hpp); this checks the WASM build agrees with it, which is where the layout
// bugs live (RGBA vs RGB, HEAP32 vs HEAPF32, pointer returned per call vs cached).
//
//   node test_lpr_box.js <frame.rgba> <w> <h> <x0> <y0> <x1> <y1> [expected-region-index]
//
// Make the .rgba the same way test_yolox.js's fixtures are made (see make_fixtures.py).
'use strict';
const fs = require('fs');
const createLPR = require('./lpr.js');

const [raw, W, H, x0, y0, x1, y1, wantRegion] = process.argv.slice(2);
if (!raw) { console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(0, 13).join('\n')); process.exit(1); }

createLPR().then(M => {
  M.FS.mkdir('/lpr');
  M.FS.writeFile('/lpr/manifest.txt', fs.readFileSync('../pure/ref/manifest.txt'));
  M.FS.writeFile('/lpr/weights.bin', fs.readFileSync('../pure/ref/weights.bin'));
  M._fn_ready();

  const buf = fs.readFileSync(raw);
  if (buf.length !== W * H * 4) throw new Error(`${raw} is ${buf.length} bytes, expected ${W * H * 4} for ${W}x${H} RGBA`);
  const p = M._malloc(buf.length); M.HEAPU8.set(buf, p);
  const t = Date.now();
  const op = M._fn_recognize_box(p, +W, +H, +x0, +y0, +x1, +y1) >> 2;
  const ms = Date.now() - t;
  const arg = [], conf = [];
  const cp = M._fn_conf() >> 2;
  for (let i = 0; i < 9; i++) { arg.push(M.HEAP32[op + i]); conf.push(M.HEAPF32[cp + i]); }
  M._free(p);

  console.log(`box (${x0},${y0})-(${x1},${y1})  ${ms} ms`);
  console.log(`  argmax   ${arg.join(' ')}`);
  console.log(`  agreement ${conf.map(v => v.toFixed(2)).join(' ')}`);

  let fails = 0;
  const ok = (c, what, extra = '') => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${what}${extra ? '   ' + extra : ''}`); if (!c) fails++; };
  ok(arg.every(v => v >= 0 && v < 200), 'argmax indices are in range');
  // Every head is a share of a summed probability over 6 crops, so it cannot exceed 1 and
  // cannot be below 1/nclasses. A 0 here means the conf pointer was read as the wrong type.
  ok(conf.every(v => v > 0 && v <= 1.0001), 'agreement scores are probabilities');
  ok(conf.slice(5).every(v => v > 0.8), 'the 4 digits are read confidently', conf.slice(5).map(v => v.toFixed(2)).join(' '));
  if (wantRegion !== undefined) ok(arg[0] === +wantRegion, `region index is ${wantRegion}`, `got ${arg[0]}`);
  console.log(fails ? `\nFAILED (${fails})` : '\nall passed');
  process.exit(fails ? 1 : 0);
});
