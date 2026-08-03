#!/bin/sh
# Build the plate-DETECTOR WASM module (uses the yolox engine). Weights not baked: the page fetches
# ../yolox/plate_yolox_tiny.onnx into MEMFS.
set -e
cd "$(dirname "$0")"

# emsdk_env.sh does not always export into Git Bash on Windows; EMSDK= is the fallback, and
# the extensionless launcher is not resolvable there either.
if command -v emcc >/dev/null 2>&1; then EMCC=emcc; else
  [ -n "$EMSDK" ] || { echo "emcc not on PATH and EMSDK unset (try EMSDK=/c/prog/emsdk/emsdk $0)"; exit 1; }
  export EM_CONFIG="$EMSDK/.emscripten"
  export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"
  NODEDIR=$(ls -d "$EMSDK"/node/*/bin 2>/dev/null | head -1) && export PATH="$NODEDIR:$PATH"
  EMCC=emcc.exe
fi

$EMCC -O3 -std=c++20 -DNDEBUG -msimd128 \
  -I../yolox/pure -I../yolox/pure/third_party \
  yolox_wasm.cpp \
  -sEXPORTED_FUNCTIONS=_fn_detect,_fn_boxes,_fn_ready,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,HEAP32,FS \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB \
  -sMODULARIZE=1 -sEXPORT_NAME=createYOLOX \
  -sENVIRONMENT=web,node \
  -o yolox.js
echo "built: yolox.js yolox.wasm"
