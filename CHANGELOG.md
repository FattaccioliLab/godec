# Changelog

## [1.0.0] — 2023-02-01

### Initial release
- Python 3.7 translation of the MATLAB GreBsmo code (Zhou & Tao, AISTATS 2013).
- Supports 8-bit and 16-bit TIFF stacks and AVI movies as input.
- Outputs four TIFF stacks: Original, Background, Sparse, Noise.

## [1.1.0] — 2025

### Changed
- Migrated from `skimage.external.tifffile` (removed in scikit-image 0.18) to standalone `tifffile` package.
- Replaced `Pipfile` / `requirements.txt` with a single `pyproject.toml`; bumped minimum Python to 3.10.
- Reorganised source into a proper `godec` package (`godec/decompose.py`) with a clean public API.
- Moved example script to `examples/run_example.py`; accepts an optional CLI argument for the input file.
- Cleaned `.gitignore`: removed macOS artifacts (`__MACOSX/`), fixed corrupted line, scoped output TIF pattern to GoDec output filenames only.
- Translated remaining French inline comments to English; fixed minor typos in docstrings.
- Added `tests/test_decompose.py` with synthetic-data smoke tests.
- Removed tracked PDF (copyrighted paper); replaced with DOI reference in README.
- Large sample files (`data/`) moved to Git LFS.
