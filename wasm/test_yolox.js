const createYOLOX = require('./yolox.js'); const fs = require('fs');
createYOLOX().then(M => {
  M.FS.mkdir('/plate'); M.FS.writeFile('/plate/model.onnx', fs.readFileSync('../yolox/plate_yolox_tiny.onnx'));
  const t0=Date.now(); M._fn_ready(); console.log('onnx parsed in', Date.now()-t0,'ms');
  const w=416,h=416; const rgba=new Uint8Array(w*h*4).fill(128);   // gray frame
  const ptr=M._malloc(rgba.length); M.HEAPU8.set(rgba,ptr);
  const t1=Date.now(); const n=M._fn_detect(ptr,w,h,30); console.log('fn_detect ->',n,'plate boxes in',Date.now()-t1,'ms');
  M._free(ptr);
  console.log('OK — detector runs end to end in WASM');
});
