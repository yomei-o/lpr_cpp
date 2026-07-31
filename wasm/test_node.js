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
  console.log('WASM argmax:', JSON.stringify(o));
  console.log('ref  argmax: [93,5,0,0,9,10,4,2,2]');
  console.log(JSON.stringify(o)===JSON.stringify([93,5,0,0,9,10,4,2,2]) ? 'MATCH' : 'MISMATCH');
});
