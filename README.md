# Low-Rank Sparse Noise Decomposition — GoDec

Python implementation of the **Greedy Semi-Soft GoDec** algorithm (GreBsmo) by Tianyi Zhou and Dacheng Tao, originally published at AISTATS 2013. The algorithm decomposes an image sequence into three components: a low-rank background, a sparse foreground, and a residual noise term. It is well-suited for background subtraction in fluorescence microscopy or any video where the background is slowly varying.

Original MATLAB code: https://tianyizhou.wordpress.com/2014/05/23/aistats-2013-grebsmo-code-is-released/

## Input / Output

The function accepts **8-bit or 16-bit TIFF stacks** and **grayscale or RGB `.avi` movies** (RGB is converted to grayscale). It produces four TIFF stacks in the current directory:

| File | Content |
|---|---|
| `1-Original.tif` | Original stack (or grayscale version of the `.avi`) |
| `2-Background.tif` | Low-rank component (background) |
| `3-Sparse.tif` | Sparse component (moving objects) |
| `4-Noise.tif` | Residual noise |

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/FattaccioliLab/Codes
cd Codes/LowRankSparseNoiseDecomposition-GoDec
pip install .
```

For a development install with linting and test dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### From the command line

```bash
python examples/run_example.py data/Escalator-1000f-8b.tif
```

### From Python

```python
from godec import DecomposeGoDec

# Default parameters, all frames
DecomposeGoDec("path/to/stack.tif")

# Custom parameters, first 500 frames only
DecomposeGoDec(
    "path/to/movie.avi",
    rank=3,
    power=5,
    tau=7,
    tol=0.001,
    k=2,
    dynamicrange=16,
    max_frames=500,
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `filename` | — | Path to `.tif`, `.tiff`, or `.avi` |
| `rank` | `3` | Upper bound on rank of the low-rank component |
| `power` | `5` | Power scheme order (higher → better accuracy, slower) |
| `tau` | `7` | Soft thresholding value for the sparse component |
| `tol` | `0.001` | Convergence tolerance (relative residual norm) |
| `k` | `2` | Rank step size |
| `dynamicrange` | `8` | Output bit depth (8 or 16) |
| `max_frames` | `0` | Frames to process (0 = all) |

Note: `rank`, `tau`, `power`, `tol`, and `k` are algorithm parameters inherited from Zhou & Tao. For most microscopy data, the defaults are a reasonable starting point; `tau` may need tuning depending on signal intensity.

## Sample data

The `data/` folder contains two test sequences:

- `Escalator-1000f-8b.tif` — first 1000 frames of the CDnet escalator sequence, converted to an 8-bit TIFF stack.
- `Escalator.avi` — the original AVI movie.

These files are large and tracked via [Git LFS](https://git-lfs.github.com/). Run `git lfs pull` after cloning if they are not present.

For additional benchmark video sequences commonly used in the background subtraction community, see the [CDnet 2014 dataset](http://changedetection.net/) and the [Background Models Challenge (BMC)](http://bmc.iut-auvergne.com/).

## Running tests

```bash
pytest tests/
```

## References

Tianyi Zhou and Dacheng Tao, *GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy Case*, ICML 2011.

Tianyi Zhou and Dacheng Tao, *Greedy Bilateral Sketch, Completion and Smoothing*, AISTATS 2013. DOI: [10.5555/3042817.3042875](https://dl.acm.org/doi/10.5555/3042817.3042875)

Python implementation: J. Fattaccioli (Department of Chemistry, ENS) — March 2020.
