#!/bin/sh
# Build the LPR WASM module (needs Emscripten on PATH). Produces lpr.js + lpr.wasm. Weights are NOT
# baked in — the page fetches pure/ref/{manifest.txt,weights.bin} and writes them into MEMFS.
set -e
cd "$(dirname "$0")"
emcc.exe -O3 -std=c++20 -DNDEBUG -msimd128 \
  -I../pure -I../pure/third_party \
  lpr_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_recognize,_fn_recognize_chw,_fn_recognize_box,_fn_conf,_fn_ready,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,HEAP32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=32MB \
  -sMODULARIZE=1 -sEXPORT_NAME=createLPR \
  -sENVIRONMENT=web,node \
  -o lpr.js
echo "built: lpr.js lpr.wasm (weights loaded at runtime from pure/ref/)"
