// Detector smoke test for the WASM build.
//
// The previous version fed a flat grey frame and asserted 0 boxes. That passes whether or not
// the detector works — and it did pass, while the decode was double-sigmoiding the ONNX head
// and every real frame came back as a scatter of boxes all sitting at ~0.30. A test that only
// checks the negative case cannot see that.
//
// So this feeds real photos with a plate in a known place and asserts the box lands on it,
// that the score is decisive rather than pinned at the threshold, and that there is no scatter.
//
//   node test_yolox.js [dir-with-fixtures]        default: ./testdata
//
// Fixtures are raw RGBA dumps, so the test needs no image decoder:
//   python -c "from PIL import Image; import numpy as np; im=Image.open('x.jpg').convert('RGBA'); \
//              np.asarray(im,dtype=np.uint8).tofile('x.rgba'); open('x.dim','w').write(f'{im.width} {im.height}')"
'use strict';
const fs = require('fs');
const path = require('path');
const createYOLOX = require('./yolox.js');

// name -> where the plate actually is, in image pixels [x1,y1,x2,y2]
const CASES = [
  { name: 'jp1', want: [682, 401, 783, 477], tol: 40 },
  { name: 'jp2', want: [249, 406, 337, 464], tol: 40 },
];

let fails = 0;
const ok = (c, what, extra = '') => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${what}${extra ? '   ' + extra : ''}`); if (!c) fails++; };

createYOLOX().then(M => {
  M.FS.mkdir('/plate');
  M.FS.writeFile('/plate/model.onnx', fs.readFileSync('../yolox/plate_yolox_tiny.onnx'));
  const t0 = Date.now();
  M._fn_ready();
  console.log(`onnx parsed in ${Date.now() - t0} ms\n`);

  // ---- negative case: a flat grey frame must find nothing
  {
    const w = 416, h = 416, rgba = new Uint8Array(w * h * 4).fill(128);
    const p = M._malloc(rgba.length); M.HEAPU8.set(rgba, p);
    const n = M._fn_detect(p, w, h, 30);
    M._free(p);
    ok(n === 0, 'flat grey frame -> no plates', `${n} found`);
  }

  // ---- positive cases: the half the old test was missing
  const dir = process.argv[2] || path.join(__dirname, 'testdata');
  let ran = 0;
  for (const c of CASES) {
    const raw = path.join(dir, c.name + '.rgba'), dim = path.join(dir, c.name + '.dim');
    if (!fs.existsSync(raw) || !fs.existsSync(dim)) { console.log(`  skip ${c.name} (no fixture in ${dir})`); continue; }
    ran++;
    const [w, h] = fs.readFileSync(dim, 'utf8').trim().split(/\s+/).map(Number);
    const buf = fs.readFileSync(raw);
    const p = M._malloc(buf.length); M.HEAPU8.set(buf, p);
    const t = Date.now();
    const n = M._fn_detect(p, w, h, 25);
    const ms = Date.now() - t;
    const bp = M._fn_boxes() >> 2;
    const boxes = [];
    for (let i = 0; i < n; i++)
      boxes.push([M.HEAPF32[bp + i * 5], M.HEAPF32[bp + i * 5 + 1],
                  M.HEAPF32[bp + i * 5 + 2], M.HEAPF32[bp + i * 5 + 3], M.HEAPF32[bp + i * 5 + 4]]);
    M._free(p);

    console.log(`\n  ${c.name} (${w}x${h}) -> ${n} plate(s) in ${ms} ms`);
    for (const b of boxes)
      console.log(`     (${b[0].toFixed(0)},${b[1].toFixed(0)})-(${b[2].toFixed(0)},${b[3].toFixed(0)})  score ${b[4].toFixed(3)}`);
    ok(n >= 1, `${c.name}: a plate is found`);
    if (!n) continue;

    const best = boxes.reduce((a, b) => (b[4] > a[4] ? b : a));
    const off = Math.max(...best.slice(0, 4).map((v, i) => Math.abs(v - c.want[i])));
    ok(off <= c.tol, `${c.name}: box is on the plate`, `worst corner off by ${off.toFixed(0)} px`);
    // With the double sigmoid every score sat just above the cut; a real detection is decisive.
    ok(best[4] > 0.5, `${c.name}: score is decisive`, best[4].toFixed(3));
    ok(n <= 3, `${c.name}: not a scatter of boxes`, `${n} found`);
  }
  if (!ran) {
    console.log('\n  NOTE: no fixtures found, so only the negative case ran — which is exactly the');
    console.log('        blind spot this test exists to close. See the header for how to make them.');
  }
  console.log(`\n${fails ? `FAILED (${fails})` : 'all passed'}`);
  process.exit(fails ? 1 : 0);
});
