const createLPR = require('./lpr.js'); const fs = require('fs');
createLPR().then(M => {
  M.FS.mkdir('/lpr');
  M.FS.writeFile('/lpr/manifest.txt', fs.readFileSync('../pure/ref/manifest.txt'));
  M.FS.writeFile('/lpr/weights.bin', fs.readFileSync('../pure/ref/weights.bin'));
  M._fn_ready();
  const b = fs.readFileSync('../pure/ref/input.bin'); const chw = new Float32Array(b.buffer, b.byteOffset, b.length/4);
  const ptr = M._malloc(chw.length*4); M.HEAPF32.set(chw, ptr>>2);
  const outPtr = M._fn_recognize_chw(ptr); const o=[]; for(let i=0;i<9;i++) o.push(M.HEAP32[(outPtr>>2)+i]);
  M._free(ptr);
  // expected argmax per head from the parity ref written next to the weights (refout.bin)
  const DIMS = [133, 10, 20, 22, 53, 11, 11, 11, 10];
  const rb = fs.readFileSync('../pure/ref/refout.bin'); const r = new Float32Array(rb.buffer, rb.byteOffset, rb.length/4);
  const ref = []; let off = 0;
  for (const d of DIMS) { let k = 0; for (let i = 1; i < d; i++) if (r[off+i] > r[off+k]) k = i; ref.push(k); off += d; }
  console.log('WASM argmax:', JSON.stringify(o));
  console.log('ref  argmax:', JSON.stringify(ref));
  const ok = JSON.stringify(o) === JSON.stringify(ref);
  console.log(ok ? 'MATCH' : 'MISMATCH'); if (!ok) process.exitCode = 1;
});
