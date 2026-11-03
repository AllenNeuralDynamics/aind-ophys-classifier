# aind-ophys-classifier

Classifies extracted ophys ROIs as soma and/or dendrite, using a ROICaT ROInet
embedding followed by two ONNX logistic-regression classifiers, and flags ROIs
touching the field-of-view border.

This capsule is a **thin wrapper**: all logic lives in the
[`aind-ophys-classifier-library`](https://github.com/AllenNeuralDynamics/aind-ophys-classifier-library)
package, and `code/run_capsule.py` only parses settings and calls
`aind_ophys_classifier_library.job.run`.

In the pipeline this capsule is **flattened** — `main.nf` feeds it
`extraction.out.capsule_results.flatten()`, so one task handles one plane.

## Input

Parameters are a `pydantic-settings` model (`ClassifierSettings` in the
library) and are passed as `python run_capsule.py --name=value`, which is the
only form a Code Ocean app panel emits. **Hyphenated flags are not accepted** —
the v1 argparse parser took `--input-dir`, `--border-size` and so on, and those
now have to be underscored.

| parameter | default | meaning |
|---|---|---|
| `--input_dir` | `/data` | mounted extraction results |
| `--output_dir` | `/results` | where per-plane outputs are written |
| `--tmp_dir` | `/scratch` | scratch space; must be writable |
| `--soma_classifier_path` | `/data/2p_roi_classifier/soma.classification_training.autoclassifier.onnx` | soma ONNX model |
| `--dendrite_classifier_path` | `/data/2p_roi_classifier/dendrite.classification_training.autoclassifier.onnx` | dendrite ONNX model |
| `--model_name` | *(unset)* | identifier recorded in `processing.json`; defaults to the soma model's parent directory name |
| `--border_size` | `10` | pixels from the FOV edge for an ROI to count as a border ROI |
| `--um_per_pixel` | *(unset)* | override for micrometres per pixel; read from the acquisition metadata when omitted |
| `--roinet_dir` | *(unset)* | directory holding a pre-staged `ROInet.zip` |
| `--verify` | `false` | round-trip the emitted `processing.json` and `quality_control.json` back through the core aind-data-schema v2 objects |

### `um_per_pixel` is load-bearing

It scales the ROI images fed to ROInet, so it changes the embedding and
therefore every prediction. It is read from `session.json` (v1
`fov_scale_factor`) or `acquisition.json` (v2 — the `Scale` entry of
`PlanarImage.image_to_acquisition_transform`). If neither carries it the run
**fails** rather than defaulting, because ROICaT silently assumes 1.0 µm/pixel
when the value is missing. Supply `--um_per_pixel` only when the metadata
genuinely lacks it; the resolved value and its source are recorded in
`processing.json`.

### Runtime assets

- The **ONNX classifiers** arrive as a mounted Code Ocean data asset
  (`2p_roi_classifier`); both are checked for existence before any work starts,
  and their MD5s are recorded in `processing.json`.
- The **ROInet weights** (`ROInet.zip`, ~105 MB) cannot be committed under
  `code/` — the archive is larger than GitHub's 100 MiB per-file limit. `code/run`
  looks for a pre-staged copy in `../data/2p_ophys_classifier`,
  `../data/2p_roi_classifier` or `../data/roinet` and, when it finds one, copies
  it to a writable scratch directory and passes `--roinet_dir` so ROICaT loads it
  locally instead of downloading. With no staged copy it warns and lets ROICaT
  fetch the archive, which inside the pipeline means one download per task.

## Output

Per plane, under `results/<plane>/classification/`:

- `<plane>_classification.h5` — groups `soma` and `dendrites` (each with
  `predictions` and `probabilities`) plus `border/labels`. These names are the
  contract with `aind-ophys-nwb`, which pairs ROIs **by positional index** in
  suite2p order.
- five overlay figures, plus `<plane>_combined_classifications.png`
- a full `processing.json` and a full `quality_control.json`

## License

See the [LICENSE](LICENSE) file for details.

## Authors

Developed by the Allen Institute for Neural Dynamics (AIND) team.
