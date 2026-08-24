# Completed Tasks and Decisions Log

This document records all the steps, changes, and architectural decisions made across recent optimization sessions for FaceFusion.

## 1. Webcam Stream Bottleneck Elimination

**Goal:** Eliminate rendering lag, micro-stutters, and high memory churn during webcam streaming.

### Changes Made:
- **Processor Pre-validation:** Moved `processor_module.pre_process('stream')` calls to the initialization phase in `multi_process_capture` (in `streamer.py`).
  - *Decision:* Removed 30-60 disk image reads and ONNX face detections per second by doing it once up front.
- **Pre-loaded Module References:** Passed pre-resolved `processor_modules` directly into `process_stream_frame`.
  - *Decision:* Eliminated Python GIL lock contention and dynamic `importlib` resolutions that were happening per-frame.
- **Static Embedding Caching:** Implemented `SOURCE_EMBEDDING_CACHE` in `face_swapper/core.py` (`prepare_source_embedding()`).
  - *Decision:* Static reference photo embeddings are calculated once per session instead of redundantly running `numpy.dot()` matrix multiplications for every detected face every frame.
- **Decoupled Camera Capture:** Created a daemon `CameraCaptureThread` in `streamer.py` using a non-blocking queue.
  - *Decision:* The main thread was previously stalling for up to 16ms waiting for hardware V-sync via `camera_capture.read()`. The background thread allows the main rendering loop to yield frames instantaneously.
- **Decoupled NSFW Analysis:** Moved the `analyse_stream()` safety check into `analyse_stream_background` within the `ThreadPoolExecutor`.
  - *Decision:* The ONNX safety check runs once per second, but previously blocked the main loop for ~50-80ms, causing a visible stutter. It is now a fire-and-forget background task.
- **Optimized Array Resizing:** Added early return conditionals in `fit_contain_frame` and `fit_cover_frame` within `vision.py`.
  - *Decision:* Bypassed `cv2.resize()` and `numpy.pad()` matrix copies when the incoming resolution perfectly matches the target, saving hundreds of megabytes in garbage-collected memory allocations per second.

---

## 2. Shared Webcam Capture (OBS Support) — REVERTED

**Goal:** Allow other applications like OBS Studio to record the physical webcam simultaneously alongside FaceFusion without receiving a black screen.

### Changes Made (then reverted):
- **Media Foundation Backend:** Updated `get_local_camera_capture()` in `camera_manager.py` to explicitly pass `cv2.CAP_MSMF` when initializing `cv2.VideoCapture` on Windows devices.
  - *Decision:* By default, OpenCV uses DirectShow (`CAP_DSHOW`) which demands an exclusive hardware lock. Media Foundation routes the request through the Windows Camera Frame Server, which natively splits the video feed for multiple applications to access simultaneously.
- *Status:* Shipped in `8f81f41` ("webcam recording") and reverted the same day in `803b62a` — forcing `cv2.CAP_MSMF` broke camera capture outright (some devices/drivers don't expose a working MSMF backend, so `VideoCapture` failed to open). `camera_manager.py` is currently back to the plain `cv2.VideoCapture(camera_id)` call with no backend override. OBS shared-capture support is **not present** in the current code; this section had gone stale until this update.

---

## 3. Initial Model Loading & Memory Optimization

**Goal:** Drastically speed up the initial startup times and context switching by optimizing how ONNXRuntime loads `.onnx` models from disk into VRAM.

### Changes Made:
- **ONNX Model Bytes Caching:** Implemented `MODEL_BYTES_CACHE` dictionary in `inference_manager.py`.
  - *Decision:* Instead of passing file paths to `InferenceSession` (which forces disk I/O every time), the `.onnx` file is read into RAM once as raw bytes. Any subsequent initializations across multiple GPUs, background threads, or contexts instantly read the model from shared memory.

---

## 4. VRAM Allocation Race Condition Fix

**Goal:** Fix the issue where VRAM allocation crawled at ~100MB/s during startup when using multiple execution threads.

### Changes Made:
- **Thread-Safe Initialization Lock:** Added `INFERENCE_LOCK = threading.Lock()` around the model initialization and memory arena allocation blocks inside `get_inference_pool()` (`inference_manager.py`).
  - *Decision:* Discovered a catastrophic race condition where all background execution threads (e.g., 8 threads) were attempting to initialize the model into VRAM and run CuDNN benchmarks at the exact same millisecond.
  - *Decision:* Retained the CuDNN `EXHAUSTIVE` benchmark flag to ensure maximum runtime FPS, but used the lock to force the benchmark to run cleanly only **once**. The waiting threads instantly receive the shared pointer once the first thread finishes.

---

## 5. Streaming Hot-Path Cleanup (follow-up to §1 and §4)

**Goal:** A code review against this log found that a couple of the "done" streaming optimizations were not actually delivering their claimed benefit in practice, plus one new lock-contention issue introduced by §4. Fixed all four.

### Changes Made:
- **Fixed the broken source-embedding cache:** `SOURCE_EMBEDDING_CACHE` in `face_swapper/core.py` was keyed on `id(source_face)`. `process_frame()` calls `extract_source_face()` fresh every frame, which builds a brand-new `Face` namedtuple each time via `average_face_identity()` — so the `id()` changed (almost) every frame and the cache never actually hit. Worse, on `ghost`/`hififace`/`simswap` models this meant an extra `embedding_converter` ONNX forward pass was silently re-run every frame instead of once per session, and the dict grew unbounded.
  - *Fix:* re-keyed the cache on `(source_face.embedding.tobytes(), face_swapper_model)` — a content-stable key — so the embedding (and conversion pass) is genuinely computed once per source face/model and reused for the rest of the session.
- **Cut redundant per-frame face-cache hashing by ~4x:** `get_static_faces()` (`face_creator.py`) called into `face_store.py`'s `get_faces()` / `resolve_lock()` / `set_faces()`, each of which independently recomputed `zlib.crc32(vision_frame.tobytes())` over the full frame buffer — up to 4 hashes of a multi-megabyte buffer per frame just to do 1-3 dict lookups.
  - *Fix:* added `face_store.create_vision_hash()`, computed once per frame in `get_static_faces()`, and threaded that hash through `get_faces`/`resolve_lock`/`set_faces` instead of the raw frame. Cache semantics (including the useful case of multiple processors in one pipeline sharing one frame's face detection) are unchanged; only the redundant hashing was removed.
- **Removed the now-warm-path lock contention from §4's fix:** `get_inference_pool()` unconditionally acquired the global `INFERENCE_LOCK` on every call, even though after warm-up it's just a dict lookup — and it's called for every processor module, every frame, from every thread in the streaming executor.
  - *Fix:* added `is_inference_pool_cold()` as a lock-free pre-check; the lock (and the original §4 race-safe initialization logic, untouched) is only taken the first time a given module/device/provider combination needs to be created.
- **Stopped copying and submitting every camera frame for a check that only runs once a second:** in `streamer.py`, `capture_vision_frame.copy()` plus an `executor.submit(...)` ran on *every* captured frame; the 1-in-`fps` sampling only happened after that, inside `content_analyser.analyse_stream()`. That's ~29 wasted multi-megabyte copies + thread-pool submissions for every 1 real NSFW check.
  - *Fix:* moved the `1-in-camera_fps` sampling into the capture loop itself, so the frame is only copied and submitted on the sampled frame. Calls `content_analyser.analyse_frame()` (already public, unsampled) directly instead of `analyse_stream()`.
  - *Note:* `content_analyser.py` itself was **not modified** — `core.py`'s `common_pre_check()` hashes that module's exact source (`hash_helper.create_hash(inspect.getsource(content_analyser).encode()) == '3c6ce25e'`) and refuses to start (`hard_exit(2)`) on any mismatch. This is a deliberate anti-tamper guard on the mandatory NSFW safety check, so all four fixes above route around it rather than touching it. A request to add a toggle to disable content analysis entirely was declined for the same reason — this is a load-bearing safety control in a face-swap tool, not a performance knob.

---

## 6. Per-Stream Source Preparation

**Goal:** Remove the remaining work performed repeatedly on source images that stay unchanged for the lifetime of a webcam session.

### Changes Made:
- **Prepared source data once per stream:** Added an optional `prepare_stream_inputs()` processor hook. `multi_process_capture()` invokes it after successful processor validation and passes the resulting processor-specific inputs to every frame task.
  - *Decision:* Keeps prepared data scoped to the active stream instead of relying on another process-global cache that could retain stale source/model state.
- **Cached face-swapper source faces and averaged identity:** The face swapper now detects, sorts, and averages source faces during stream initialization.
  - *Decision:* Eliminates repeated full-image serialization and hashing, source-face sorting, and `average_face_identity()` work on every webcam frame. `select_faces()` accepts the prepared source faces while retaining its existing fallback for non-stream workflows.
- **Prepared model source input once:** Embedding-based models prepare the source embedding once; `blendswap` and `uniface` prepare their warped source frame once. These immutable values are reused by all webcam frame workers.
  - *Decision:* Removes source-only preprocessing from the concurrent frame hot path without changing target-dependent embedding balancing.
- **Restored processor validation semantics:** Stream initialization now retains only processors whose `pre_process('stream')` call succeeds.
  - *Fix:* The earlier pre-validation move discarded the Boolean result and could execute a processor even after its validation failed. Validation still occurs only once, but failed processors are no longer submitted for frame processing.

---

## 7. Reusable Webcam Vision Mask

**Goal:** Eliminate a large, constant allocation performed for every ordinary webcam frame.

### Changes Made:
- **Reused the default opaque mask:** The capture loop creates one read-only, all-white vision mask from the first three-channel camera frame and passes it to every frame-processing task.
  - *Decision:* Avoids allocating and initializing a full-resolution mask for every processed frame (about 2 MB per frame at 1080p) while keeping concurrent workers safe from accidental in-place mutation.
- **Preserved dynamic-format handling:** The mask is recreated if the camera resolution changes, while alpha-channel frames and unexpected mask mismatches fall back to `extract_vision_mask()`.
  - *Decision:* Retains the existing behavior for nonstandard frames and leaves all non-webcam workflows unchanged.

---

## 8. Demand-Driven Analysis and Low-Latency Scheduling

**Goal:** Avoid unused face-model inference and small per-frame allocations while reducing webcam latency caused by slow, stale frame tasks.

### Changes Made:
- **Frame-scoped target-face reuse:** The first face processor in a webcam frame performs target detection or tracking, and later processors reuse the same target faces through a worker-local context.
  - *Decision:* Removes repeated full-frame hashing and cache synchronization for multi-processor pipelines, avoids advancing the face tracker more than once for a single captured frame, and keeps non-webcam workflows unchanged.
- **Demand-driven face attributes:** Webcam workers now request target embeddings only for reference selection and face-swapper models that blend with the target identity. Demographic classification runs only for age modification or active gender, race, or narrowed age filters.
  - *Decision:* Skips complete face-recognizer and face-classifier ONNX passes when their outputs are unused. Full analysis remains the default outside webcam workers, and reduced-analysis face-cache entries use a separate cache scope.
- **Adaptive stale-result dropping:** Adaptive mode can display the newest completed frame without waiting for an older slow future. Older tasks are cancelled when possible and otherwise tracked until completion so they still count against the in-flight limit.
  - *Decision:* Reduces head-of-line latency without allowing discarded work to create an unbounded executor queue. Non-adaptive modes retain chronological output.
- **Reusable empty audio inputs:** Empty audio and voice arrays are allocated once per webcam session, marked read-only, and shared by frame tasks.
  - *Decision:* Removes two small allocations per processed frame while retaining fallback allocation behavior for direct callers of `process_stream_frame()`.

---

## 9. Per-Frame Overhead, Array Dtypes, and Honest Stream Metrics

**Goal:** A fresh bottleneck review of the webcam path found high-frequency overhead outside the ONNX passes themselves — repeated cache lookups that can never hit, stack walks on every state read, float64 arithmetic on frame-sized buffers — plus a scheduling gate that idled workers and an overlay that overstated throughput. Each item below is a separate commit.

*Method note:* this pass was **static analysis only** — no profiler was run and the test suite was not executed, because the review box has no runtime libraries installed. Impact estimates below are reasoned from per-frame work, not measured. Worth re-checking against a real profile before treating any ranking as settled.

### Changes Made:
- **Skipped face-store hashing for streamed target frames** (`e95e66f`): `get_static_faces()` hashed every target frame with `zlib.crc32(vision_frame.tobytes())` — a full-frame copy plus a full-frame scan (~2.7 MB at 720p) — to consult a cache that can never hit, because webcam frames are unique by construction. Each miss also allocated a `threading.Lock`, inserted a `FACE_STORE` entry, and evicted another.
  - *Fix:* added a thread-local `cache_bypass` flag in `face_creator.py`; when set, `get_static_faces()` delegates straight to `get_many_faces()`. `select_faces()` enables it only around **target**-face resolution, and only when `FACE_SELECTION_CONTEXT.enabled` (which is true exactly for the stream path). Source-face caching — where the cache genuinely does hit, since the source image is constant for the session — is untouched, as is `extract_source_face()`'s fallback. Covers the `track_faces()` branch too, since it also routes through `get_static_faces()`.
- **Cached app-context detection** (`6dc7364`): `detect_app_context()` walked the entire Python stack, substring-matching `co_filename` against `facefusion/jobs` and `facefusion/uis` on every frame — and `state_manager.get_item()` called it unconditionally. `swap_face()` alone does ~12 state reads per face, and `inference_manager.get_inference_pool()` several per ONNX run, putting this on the order of 100–200 stack walks per frame.
  - *Fix:* two layers. A module-level `FILE_CONTEXT_CACHE` memoizes the per-filename classification (code objects reuse the same `co_filename` string, so the dict hash is cached), and a thread-local override short-circuits the walk entirely. `process_stream_frame()` sets the override to `detect_app_context()`'s own result once per frame and clears it in `finally`.
  - *Decision:* the override is *computed*, not hardcoded to `'cli'`, so it stays provably identical to the old answer. A deeper call site walks through extra `facefusion/processors`, `face_creator`, etc. frames first — none of which classify — then reaches the same root frames, so the result cannot differ.
- **Cached available execution providers** (`32955fe`): `get_available_execution_providers()` had no `lru_cache` (unlike `get_onnxruntime_version()` directly above it), so it called into onnxruntime and rebuilt its list on every invocation. `has_execution_provider()` reaches it from `conditional_thread_semaphore()` (up to 3x) *and* from `get_inference_pool()` — roughly 4x per ONNX forward, 25+ times per frame.
  - *Fix:* split into a cached `detect_available_execution_providers()` returning a tuple (used by `has_execution_provider()`, now allocation-free) and a thin `get_available_execution_providers()` returning a fresh list.
  - *Decision:* the list wrapper is deliberate, not redundant. `uis/components/execution.py` stores the result into `state_manager`, so handing out the cached object itself would create a mutable-shared-state hazard.
- **Built the face-detector input as float32** (`a5dadf9`): `prepare_detect_frame()` allocated `numpy.zeros((640, 640, 3))` — float64, ~9.8 MB zeroed — filled it, then transposed (strided) and `.astype(float32)`. `normalize_detect_frame()` then added two more full-array temporaries. Roughly 25 MB of allocation and 4–5 full passes per detector call.
  - *Fix:* allocate `(1, 3, H, W)` float32 directly and write the transposed source into the corner in one pass; normalize in place with `numpy.subtract`/`numpy.divide` and `out=`.
  - *Decision:* `normalize_detect_frame()` now mutates its argument. All four callers pass a frame straight from `prepare_detect_frame()` and own it exclusively; `content_analyser.py` has its own separate `prepare_detect_frame()` and never calls this one. Recorded as a comment on the function so the ownership contract is not accidentally broken later.
- **Kept face-swapper crop conversions in float32** (`bfdeaaf`): `mean`/`standard_deviation` are Python lists, so `prepare_crop_frame()` and `normalize_crop_frame()` promoted every operation to float64 — four ~1.6 MB temporaries per 256×256 crop, repeated per pixel-boost tile.
  - *Fix:* convert mean/std to float32 arrays once per call and chain the arithmetic through `out=` on a freshly owned buffer.
  - *Decision:* channel order was deliberately **not** folded into the mean/std application. `simswap` uses non-uniform per-channel values (`mean: [0.485, 0.456, 0.406]`), so reversing channels relative to them would be a silent correctness bug on that model. The cost is one extra contiguous copy, still far below the previous float64 churn.
  - *Note:* `normalize_crop_frame()` now returns float32 where the ghost/hififace/hyperswap/uniface branch previously returned float64. Downstream is `explode_pixel_boost()` → `cv2.warpAffine` → `astype(uint8)`, all fine with float32 (and better supported by OpenCV than float64). The other branch already returned float32, so this makes the two consistent.
- **Gated stream submission on unfinished work** (`daa3a44`): submission was capped by `len(futures) + len(discarded_futures) < max_queue_size`, which counted completed-but-not-yet-yielded futures against the worker budget. With ordered output and `max_queue_size == execution_thread_count`, one slow head frame filled the window and stopped submission entirely, leaving the executor idle behind it.
  - *Original fix:* counted only futures that are not `done()` and added `len(futures) < max_queue_size * 2` as a hard cap. The doubled hard cap was later found to permit roughly 500 ms of buffered video at 30 FPS and was reduced back to one queue window in §13.
  - *Correction to the review that prompted this:* the same review also claimed stream output was "gated on camera arrival." That was **wrong** — `continue` on an empty capture queue returns to the top of the loop, which re-drains, so finished frames are yielded within the 5 ms poll regardless. The submission window was the real defect; the latency claim was not.
- **Reported processed frame rate instead of delivered** (`ed05bb9`): on a skip, the capture loop re-yielded `last_processed_frame` paired with the *new* frame's `capture_time`. `PerformanceOverlay` counted that as a delivered frame, so the HUD overstated FPS **and** understated latency — precisely the two numbers someone reads while tuning skipping modes.
  - *Fix:* `multi_process_capture()` now yields `(frame, capture_time, is_duplicate)`; the overlay tracks processed and delivered timestamps separately, records latency only for real frames, and drives the history graphs from processed frames. The headline `FPS` is the processing rate; delivered rate appears as `OUT` in the HUD header.
  - *Decision:* window averages are measured against the current time rather than the newest sample, so a stalled pipeline decays towards zero instead of freezing at its last healthy reading. `tqdm` was already honest here — it never counted skips.

### Deliberately Not Done:
- **Redundant `paste_back()` full-frame copy:** `process_stream_frame()` already copies the target frame, then `paste_back()` copies it again per face per processor. Skippable for the stream path, but the function is shared with batch workflows where the copy is load-bearing; left alone pending a safe way to signal ownership.
- **Wasted target-face embedding at the default weight:** `get_stream_face_analysis_features()` requests `'embedding'` for hyperswap/inswapper/simswap/hififace, forcing a full recognizer pass per target face per frame. Its only consumer is `balance_source_embedding()`, where the default `face_swapper_weight` of `0.5` maps to a blend factor of exactly `0.0` — so the target embedding is multiplied by zero. Gating the feature on `face_swapper_weight != 0.5` would remove one of roughly six ONNX forwards per frame. Identified but out of scope for this pass.
- **Gradio inline delivery cost:** in `inline` mode, `webcam.py` does a full-frame `cvtColor` plus resize, then Gradio JPEG-encodes and base64s every frame over the websocket. This is plausibly the dominant end-to-end cost in that mode and is invisible to every timing the pipeline reports. Not addressed; flagged for a future pass.

---

## 10. Off-Frame Webcam Paste Safety

**Goal:** Fix the webcam face-swap crash in `error.md` where `paste_back()` attempted to blend an empty destination slice with a 256×256 transformed crop.

### Changes Made:
- **Derived the paste size from the actual destination slice:** `paste_back()` now creates the destination view before warping the crop and uses that view's real height and width for both inverse warps.
  - *Decision:* Keeps the transformed mask and frame dimensions aligned with the pixels that can actually be written, including faces partially clipped by a webcam-frame boundary.
- **Handled fully off-frame faces as a no-op:** If clipping produces an empty destination region, `paste_back()` returns the unchanged frame instead of invoking OpenCV or NumPy with incompatible dimensions.
  - *Decision:* Preserves both existing ownership modes: normal calls still return a copy, while stream-owned frames remain in place.
- **Added a regression test:** `test_paste_back_ignores_crop_outside_frame()` covers a crop whose affine transform places it completely beyond the destination frame.

### Verification:
- `git diff --check` passes.
- Python compilation passes under `python3`.
- The focused pytest suite was not run in this workspace because its current Python environment does not provide `pytest` or `numpy`; run `pytest -q tests/test_paste_context.py tests/test_face_enhancer_blend.py` in the FaceFusion Conda environment.

---

## 11. Transient Windows Webcam Read Recovery

**Goal:** Prevent the webcam stream from freezing or stopping after OpenCV's MSMF backend reports a temporary `can't grab frame` error.

### Changes Made:
- **Retried transient capture failures:** `CameraCaptureThread` now tolerates up to 30 consecutive failed reads with a 10 ms delay, instead of terminating the stream after the first failure.
  - *Decision:* A successful read resets the failure count, allowing brief Windows Media Foundation stalls to recover while persistent camera disconnection still stops the capture thread after a bounded interval.
- **Added a regression test:** `test_camera_capture_thread_recovers_from_transient_read_failure()` verifies that a valid frame following a failed read is delivered normally.

### Verification:
- Python syntax compilation and `git diff --check` pass.
- The focused pytest suite remains unavailable in this workspace because its Python environment does not provide `pytest` or `numpy`.

---

## 12. Live Webcam Processor Activation

**Goal:** Make Face Enhancer output visible when the module is enabled while the webcam is already streaming.

### Changes Made:
- **Refreshed the active processor pipeline:** The capture loop now detects changes to the configured processor names and rebuilds the validated processor-module list used for subsequent frames.
  - *Cause:* The webcam previously captured the processor list only once at startup. Enabling Face Enhancer changed the UI state, but the running stream continued submitting frames only to the previously active Face Swapper.
- **Centralized stream processor preparation:** Initial startup and live refresh now share `prepare_stream_processors()`, including `pre_process('stream')` validation and optional per-stream input preparation.
- **Added a regression test:** The new test verifies that a newly selected Face Enhancer module is validated, included, and has its stream inputs prepared.

### Verification:
- Python syntax compilation and `git diff --check` pass.
- The focused pytest suite remains unavailable in this workspace because its Python environment does not provide `pytest` or `numpy`.

## 13. Webcam Ordered-Buffer Latency Cap

**Goal:** Restore low webcam latency after the stream grew to roughly 500 ms behind live capture.

### Changes Made:
- **Reduced the ordered frame-buffer cap:** Stream submission now stops when the buffered future list reaches `execution_thread_count`, rather than allowing `2 × execution_thread_count` entries.
  - *Cause:* With the default eight threads, the previous 16-frame cap represented about 533 ms at 30 FPS. A slow oldest future could therefore retain half a second of already captured video even while newer work completed.
  - *Decision:* Unfinished-work counting is retained so completed futures do not falsely occupy executor workers, but completed frames waiting behind an ordered head now count against the latency window.
- **Added capacity-policy regression coverage:** Parameterized tests verify that submission is rejected when either active work or the ordered frame buffer reaches its configured limit.

### Verification:
- Python syntax compilation and `git diff --check` pass.
- The focused pytest suite remains unavailable in this workspace because its Python environment does not provide `pytest` or `numpy`.

---

## 14. Stable Low-Latency Webcam Default

**Goal:** Prevent the webcam delay from spending most of its time near 300 ms and oscillating between approximately 150 and 300 ms under variable processing load.

### Changes Made:
- **Made adaptive scheduling the default:** The webcam UI now starts in `adaptive` frame-skipping mode, and both the capture scheduler and performance overlay use `adaptive` when no explicit state value exists.
  - *Cause:* Even after reducing the hard buffer cap, strict ordered mode can hold eight frames with the default thread count. At 30 FPS that is roughly 267 ms before display overhead, and the queue repeatedly fills and drains as inference time varies.
  - *Decision:* Adaptive mode preserves the newest completed result and stops admitting stale work when all workers are occupied. This trades delivery of every captured frame for stable, bounded live latency. Explicit `disabled`, `1-in-2`, and `1-in-3` modes remain available.

### Verification:
- Python syntax compilation and `git diff --check` pass.
- Runtime latency must be measured in the Windows FaceFusion environment because this workspace does not provide the project runtime dependencies or webcam hardware.
