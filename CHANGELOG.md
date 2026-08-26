# Changelog

## [unreleased]

### 🚀 Features

- Add sound-speed profile helper functions (Arctic and Munk profiles)
- Add `PyRAMResults` NamedTuple for typed model outputs
- Add four examples, including comparison of Arctic and Munk sound-speed profiles
- Revise default grid-spacing calculations

### 🐛 Bug Fixes

- Prevent mutation of user-supplied input arrays
- Remove fixed MyPy Python version to improve compatibility across supported interpreters

### ⚙️ Maintenance

- Vectorize the `profl()` method
- Remove obsolete `setup.py`
- Remove legacy PyRAM code
- Refresh examples and supporting files

### ♻️ Refactoring

- Move examples into a dedicated `examples.py` module
- Simplify PyRAM and PyRAMmp tests
- Expose `PyRAMResults` in the public API
- Add type hints throughout the codebase

### 📚 Documentation

- Add comprehensive RAM documentation and references
- Convert documentation to NumPy-style docstrings
- Document numerical parameter selection and grid sizing
- Include original RAM documentation (`readme.orig`)

### 🎨 Styling

- Apply consistent formatting with Ruff

### 📦 Build System

- Modernize packaging and development tooling
- Add optional plotting support via Matplotlib
- Add lock files for supported Python versions
- Add release automation and dynamic versioning

### 🏗️ CI/CD

- Add GitHub Actions CI workflow
- Enable Ruff, MyPy, and pytest validation
- Add dedicated formatting checks
- Optimize testing and package build jobs
- Add CI validation for the `develop` branch


## [1.3.0] - 2025-03-28

### 🐛 Bug Fixes

- Replace deprecated `numpy.complex` with `numpy.complex128`.

### ⚙️ Maintenance

- Apply minor code cleanups and maintenance updates.

### 📚 Documentation

- Refresh docstrings and project documentation.
- Update Ocean Acoustics Library (OALIB) URLs.

---

## [1.2.0] - 2019-07-08

### 🚀 Features

- Return complex acoustic pressure fields from model runs.
- Return reference sound speed (`c0`) with model outputs.

### 🐛 Bug Fixes

- Correct handling of range-dependent environments with range intervals smaller than `dr`.
- Fix output grid depth indexing so output values are reported at the correct depths.
- Ensure `outpt()` cannot index beyond output array bounds.
- Correct seabed profile depth handling relative to the deepest water-profile depth.
- Fix `PyRAMmp` so multiple batches of runs are handled correctly.

### ⚙️ Maintenance

- Improve multiprocessing test flexibility with configurable repetitions (`nrep`).
- Correct multiprocessing speed-up calculation.
- Update conda packaging recipe.

### 🚀 Multiprocessing

- Add the `PyRAMmp` multiprocessing interface for parallel model execution.

### 🧪 Testing

- Improve `PyRAMmp` test coverage and validation.

---

## [1.1.7] - 2018-11-10

### 🧪 Testing

- Improve the `PyRAMmp` regression and performance test suite.

### 🐛 Bug Fixes

- Ensure `outpt()` cannot index beyond the end of output arrays.

---

## [1.0.1] - 2017-11-30

### 🐛 Bug Fixes

- Correct handling of `dr`, `ndr`, and `ndz`.
- Fix an issue where `ndr` and `ndz` were not being applied during output generation.

### 🚀 Performance

- Add Numba JIT compilation support.
- JIT-compile the `outpt()` routine.
- Remove unused function `g`.
- Introduce the `run()` method API.

### ♻️ Refactoring

- Perform minor code refactoring and cleanup.
- Apply additional bug fixes and internal restructuring.

### ⚙️ Packaging

- Add conda build recipe.
- Update conda packaging configuration.

### 🔬 Validation

- Regenerate reference transmission loss data using a Fortran build compiled with `-O3`.