#!/bin/sh
# Build the plate-DETECTOR WASM module (uses the yolox engine). Weights not baked: the page fetches
# ../yolox/plate_yolox_tiny.onnx into MEMFS.
set -e
cd "$(dirname "$0")"
emcc -O3 -std=c++20 -DNDEBUG \
  -I../yolox/pure -I../yolox/pure/third_party \
  yolox_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_detect,_fn_boxes,_fn_ready,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,HEAP32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB \
  -sMODULARIZE=1 -sEXPORT_NAME=createYOLOX \
  -sENVIRONMENT=web,node \
  -o yolox.js
echo "built: yolox.js yolox.wasm"
