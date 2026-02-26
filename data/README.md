# Sample data

This folder contains test sequences for the GoDec decomposition.

| File | Description |
|---|---|
| `Escalator-1000f-8b.tif` | First 1000 frames of the CDnet escalator sequence, 8-bit grayscale TIFF stack (~20 MB) |
| `Escalator.avi` | Original AVI movie (~7 MB) |

These files are tracked with [Git LFS](https://git-lfs.github.com/). If they are missing after cloning, run:

```bash
git lfs pull
```

## Source

The escalator sequence originates from the **CDnet 2014** change detection benchmark dataset:
http://changedetection.net/

Wang, Y., Jodoin, P.-M., Porikli, F., Konrad, J., Benezeth, Y., & Ishwar, P. (2014).
CDnet 2014: An Expanded Change Detection Benchmark Dataset.
*CVPR Workshops*.
