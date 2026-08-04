#!/usr/bin/env python3
"""Fetch the detector test fixtures and convert them to raw RGBA for test_yolox.js.

    pip install pillow
    python make_fixtures.py            # -> wasm/testdata/{jp1,jp2}.{rgba,dim}

The images are not committed — they are other people's photographs, and the repo should not
redistribute them. They are downloaded from Wikimedia Commons on demand instead:

  jp1  "ERA Mini Turbo rear.JPG"     — a clear rear-view plate, filling ~1.5% of the frame.
                                      DETECTION ONLY: the plate is a New Zealand one
                                      (CHRISTCHURCH, HZF440), so the Japanese classifier returns
                                      nonsense on it (山口 200 あ 2540, region agreement 0.28).
                                      The detector finds it fine at 0.832, which is all this
                                      fixture is for.
  jp2  "TokyuBus H1283 rear.jpg"     — a Japanese bus plate, smaller and lower contrast.
                                      Ground truth 横浜 200 か 35-91, and the full pipeline gets
                                      it exactly right, so this one is also the demo's built-in
                                      sample — see wasm/sample/ (committed, with attribution).

Both are Commons-hosted under their own licences; see the file pages for author and terms.
They are used here only as local test input.

test_yolox.js skips its positive cases when the fixtures are absent and says so, so a clone
without Pillow or without network still runs — it just cannot check the half of the behaviour
that actually matters.
"""
import os
import sys
import urllib.request

FIXTURES = {
    "jp1": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/"
           "ERA_Mini_Turbo_rear.JPG/960px-ERA_Mini_Turbo_rear.JPG",
    "jp2": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/"
           "TokyuBus_H1283_rear.jpg/960px-TokyuBus_H1283_rear.jpg",
}
UA = {"User-Agent": "lpr_cpp test fixtures (https://github.com/yomei-o/lpr_cpp)"}


def main():
    try:
        from PIL import Image
    except ImportError:
        sys.exit("need Pillow:  pip install pillow")
    import numpy as np

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
    os.makedirs(out, exist_ok=True)
    for name, url in FIXTURES.items():
        jpg = os.path.join(out, name + ".jpg")
        if not os.path.isfile(jpg):
            print(f"downloading {name}")
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                open(jpg, "wb").write(r.read())
        im = Image.open(jpg).convert("RGBA")
        np.asarray(im, dtype=np.uint8).tofile(os.path.join(out, name + ".rgba"))
        open(os.path.join(out, name + ".dim"), "w").write(f"{im.width} {im.height}")
        print(f"  {name}: {im.width}x{im.height}")
    print(f"\nwrote {out}. Now: node test_yolox.js")


if __name__ == "__main__":
    sys.exit(main())
