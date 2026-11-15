# Working with the Data Standard

## Reading and writing data

Use `farm.standard`; never write a dataset file with plain `open()` or OpenCV.

| To | Use |
| --- | --- |
| Build or split an ID | `ids.make_sample_id(session_id, frame_index)`, `ids.split_sample_id(sample_id)` |
| Find a file | `SessionDir.for_id(session_id).file("rgb" \| "depth" \| "labels" \| "masks" \| "meta", sample_id)` |
| Iterate samples | `SessionDir(...).samples()`: meta records ordered by `rgb_timestamp_ns`, not by index |
| Images | `images.read_rgb` / `write_rgb` (RGB order, not BGR), `read_depth` / `write_depth` (uint16 mm), `read_mask` / `write_mask` |
| Boxes | `labels.Box.from_xyxy_px(...)` clips and says whether the box was truncated; `labels.write_labels` |
| Change an annotation's status | `provenance.apply(record, status, by, ...)`; it refuses transitions the standard forbids |
| May a generator write? | `provenance.generator_may_write(record)`: only `none` and `generated` |
| JSON | `jsonio.write_json` (2-space indent, final newline, atomic) |
| Predictions | `predictions.detection`, `detection_line`, `write_detections` |

Keep `labels/<sample_id>.txt` and `tracks.json` in sync: `tracks.json` refers to boxes by line index.

## Validating

```bash
farm validate src/data/datasets/<session_id>     # one session
farm validate src/data                            # every session and prediction run
farm validate --json --quiet <path>               # machine-readable, errors only
```

Exit status 0 = no errors, 1 = errors, 2 = nothing found at the path. Validate every session
before building a dataset from it, and after any tool writes into it. `docs/examples/` currently
fails only on the images its README lists as still to add.

## Changing the standard

1. Write the change and the reason; get lead approval.
2. Bump `Version` in `docs/data-standard.md` and `SCHEMA_VERSION` in `src/farm/standard/schema.py`.
3. Update the library, `docs/examples/` and the tests in the same change.

Open questions found while building the library (for the lead):

- **Restoring an exclusion made at `none`.** `none -> excluded` is allowed, and a restore goes
  back to the status before exclusion. But a `none` record must have an empty history, so there
  is no valid record after restoring to `none`. `provenance.apply` refuses it for now.
- **Re-running a generator.** Section 10.3 lets a re-run overwrite `generated` annotations, but
  `generated -> generated` is not an allowed transition. So the history cannot record the
  second run, and `generator` keeps naming the first tool version.
- **Empty label files.** Section 12 asks for a `tracks.json` entry for every label file; the
  validator also asks it for empty ones (as `[]`).
