# `engine/` — the matching engine, outside the app

This project's matching core ([`FaceEngine`](../FaceID/FaceEngine.hpp)) is **pure C++17 with zero
Apple dependencies**. That's the point of the architecture: the Swift layer owns camera / Vision /
Core ML, and the engine — L2-normalize, cosine match, multi-shot templates, JSON-free persistence,
`std::mutex` thread-safety — is portable to Android (NDK) or any desktop.

This folder proves that by building the **same** `../FaceID/FaceEngine.cpp` into a test suite and a
CLI, with nothing but a C++ compiler.

## Run

```bash
make -C engine test     # compile + run unit tests  → "PASS  22 passed, 0 failed"
make -C engine cli      # build ./engine/face_cli
./engine/face_cli faces.db demo
```

`demo` output:
```
[demo] persons=3
  query(alice ) -> alice   cos=1.000
  query(bob   ) -> bob     cos=1.000
  query(carol ) -> carol   cos=1.000
  query(strangr) -> alice   cos=0.032  (should be low)
```

## What the tests cover

`tests/test_face_engine.cpp` (no framework, just a `CHECK` macro):

- self-match cosine ≈ 1, orthogonal ≈ 0
- **multi-shot**: a second template lifts recall for a probe the first template misses
- distinct-person `count()`, `templateCount()`, `names()` ordering
- `remove()` / `rename()`
- persistence round-trip across engine instances
- thread-safety smoke (concurrent enroll + query)

## Reusing on Android

`FaceEngine.{hpp,cpp}` drop straight into an NDK module. Provide the platform's embeddings (e.g. a
TFLite ArcFace) and call `enroll` / `findBest` exactly as the iOS bridge does — the matching logic,
threshold semantics, and on-disk format are identical.
