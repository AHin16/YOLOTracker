# Bubble Analyzer Checklist

## Shared Foundations

- [x] Extract tracker-free YOLO predictor construction.
- [x] Extract shared image discovery and loading.
- [x] Add tested mask-area and ROI statistics.
- [x] Add tested Include, Exclude, and Weighted policies.

## Gas Holdup Workflow

- [x] Process an input image folder without tracker imports.
- [x] Save one segmentation overlay per frame.
- [x] Write the required per-frame CSV columns.
- [x] Emit preview/progress metrics and support stopping.

## Qt Shell

- [x] Preserve Bubble Tracking as the first tab.
- [x] Add the independent Gas Holdup page.
- [x] Persist Gas Holdup settings under a namespace.
- [x] Coordinate close behavior for both workers.

## Verification

- [x] New focused tests pass.
- [x] Existing test suite passes.
- [x] Changed Python modules compile.
- [x] Offscreen Qt smoke test shows both tabs.
