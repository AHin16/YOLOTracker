# Implementation Plan: Bubble Analyzer Gas Holdup Workflow

## Overview

Convert the Qt application into a two-tab Bubble Analyzer while preserving the existing Bubble Tracking page and adding an independent Gas Holdup workflow. Both workflows use shared YOLO predictor construction and image-loading modules, while only Bubble Tracking imports tracking code.

## Architecture Decisions

- Keep the loaded tracking central widget intact and host it as the first page of a new `QTabWidget`.
- Put predictor construction and image discovery/loading in tracker-free shared modules.
- Model gas-holdup calculations as pure mask operations so ROI, filtering, edge policy, and statistics can be unit tested without YOLO or Qt.
- Treat gas holdup as the union of accepted segmentation-mask pixels divided by ROI pixels; never use bounding-box area.
- Report equivalent bubble diameter in pixels because the requested Gas Holdup parameter set has no physical calibration input.
- Preserve the existing flat tracking state keys and add Gas Holdup settings under a namespaced `gas_holdup` object.

## Task List

### Phase 1: Shared Foundations

- [x] Task 1: Extract shared predictor and image I/O modules.
- [x] Task 2: Add pure gas-holdup mask analysis with ROI and edge policies.

### Checkpoint: Foundation

- [x] Unit tests prove union-area, minimum-area, ROI, and border handling.
- [x] Existing tracking imports and tests remain valid.

### Phase 2: Processing Workflow

- [x] Task 3: Add folder processing, overlay export, CSV output, progress, and cancellation.
- [x] Task 4: Add a worker-backed Gas Holdup Qt page with files, parameters, results, and preview.

### Checkpoint: Processing

- [x] A fake predictor processes an image folder end-to-end.
- [x] Output contains one overlay and one CSV row per processed image.

### Phase 3: Application Shell

- [x] Task 5: Wrap the existing tracking UI in a Bubble Tracking tab and add Gas Holdup.
- [x] Task 6: Persist namespaced Gas Holdup state and coordinate shutdown safely.

### Checkpoint: Complete

- [x] All automated tests pass.
- [x] Python compilation succeeds for changed modules.
- [x] The Qt window opens offscreen with both workflow tabs.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tracking regression from refactor | High | Limit tracking changes to shared imports and run existing tests. |
| Overlapping instance masks inflate gas holdup | High | Calculate numerator from the union mask, not summed instance areas. |
| Edge policy ambiguity | Medium | Exclude removes frame-border masks; Include and Weighted retain only observed mask pixels, which is the measurable quantity. |
| Large TIFF inputs | Medium | Reuse normalized OpenCV/Pillow loading and emit previews at a configurable stride. |
| UI worker shutdown race | Medium | Use the existing stop-event/QThread pattern for the independent worker. |

## Open Questions Resolved by Conservative Defaults

- ROI is entered as X, Y, Width, Height; zero width or height extends to the frame boundary.
- Gas Holdup equals Bubble Area Ratio until a separate calibration model is specified.
- Average diameter is the mean equivalent-circle diameter in pixels.
