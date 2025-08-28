# blazemetrics: Release and Publishing Guide

This guide documents how to push the project to GitHub and publish releases to PyPI. It includes versioning, building, and recommended automation.

## 1) Prerequisites

- Python 3.8+
- Rust toolchain with `cargo` (install via https://rustup.rs/)
- Tools: `maturin`, `twine`, `setuptools`, `wheel`
- PyPI account and an API token
- GitHub account (`2796gaurav`) and repository `blazemetrics`

Install tools:
```powershell
python -m pip install --upgrade pip setuptools wheel maturin twine
```

## 2) Project files and metadata

- `pyproject.toml` contains the package metadata. Update when releasing:
  - `version`
  - `authors`, `description`, `classifiers`, `project.urls`
- `LICENSE` should be present (MIT).
- `README.md` is used as the long description on PyPI.

Version bump example in `pyproject.toml`:
```toml
[project]
name = "blazemetrics"
version = "0.1.1"
```

If needed, also update `Cargo.toml` crate version to match for clarity (not required by PyPI, but recommended for consistency).

## 3) GitHub setup and push

First time only:
```powershell
# Initialize Git (if not initialized yet)
git init

git add .
git commit -m "Initial commit"

git branch -M main

git remote add origin https://github.com/2796gaurav/blazemetrics.git

git push -u origin main
```

Subsequent updates:
```powershell
git add -A
git commit -m "Your message"
git push
```

Create a release tag (recommended before publishing to PyPI):
```powershell
# Update the version in pyproject.toml first, e.g., 0.1.1
git add pyproject.toml README.md LICENSE
git commit -m "Bump version to 0.1.1"

git tag v0.1.1
git push origin v0.1.1
```

## 4) Build distributions with maturin

Clean and build wheels and sdist:
```powershell
# Optional: clean old builds
if (Test-Path dist) { Remove-Item dist -Recurse -Force }

# Build a wheel for the local platform
maturin build --release --out dist

# Build sdist (source distribution)
maturin sdist --out dist
```

Validate metadata:
```powershell
twine check dist/*
```

Note: On Linux/macOS you can add `--compatibility manylinux_2_28` as needed to produce manylinux wheels. For Windows, the default local build is fine.

## 5) Publish to PyPI

Option A: Use environment variables (recommended):
```powershell
# Set PyPI token in the current shell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-<YOUR_TOKEN>"

twine upload dist/*
```

Option B: Use `~/.pypirc` (persistent):
```ini
# Windows path: C:\Users\<you>\.pypirc
[pypi]
  username = __token__
  password = pypi-<YOUR_TOKEN>
```
Then simply run:
```powershell
twine upload dist/*
```

After a successful upload, verify:
- PyPI project page: https://pypi.org/project/blazemetrics/

## 6) Sanity checklist before publishing

- Version updated in `pyproject.toml`
- `README.md` badges/links point to `2796gaurav/blazemetrics`
- `LICENSE` present and correct year/name
- `twine check dist/*` passes
- Wheel installs and imports locally:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --no-cache-dir dist\blazemetrics-<version>-*.whl
python -c "import blazemetrics as bm; print(bm.__doc__[:60])"
```

## 7) Optional: GitHub Actions for automated builds/publish

Save as `.github/workflows/release.yml` to build on tag pushes and publish to PyPI (set `PYPI_API_TOKEN` in repo secrets):
```yaml
name: Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: dtolnay/rust-toolchain@stable
      - run: python -m pip install --upgrade pip setuptools wheel maturin twine
      - run: maturin build --release --compatibility manylinux_2_28 --out dist
      - run: maturin sdist --out dist
      - run: twine check dist/*
      - name: Publish to PyPI
        if: startsWith(github.ref, 'refs/tags/')
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

## 8) Troubleshooting

- Missing Rust toolchain: install from https://rustup.rs/
- Windows PowerShell errors with `&&` or pipes: run commands separately.
- `ImportError` after install: ensure the wheel was built for your Python version and architecture.
- Unicode/README rendering on PyPI: run `twine check` to validate.

## 9) Release flow summary

1. Bump version in `pyproject.toml`
2. Commit and tag `vX.Y.Z`
3. Build wheels + sdist with `maturin`
4. `twine check` then `twine upload dist/*`
5. Verify on PyPI and create a GitHub Release (optional)

That's it! This doc should be all you need to ship new versions to PyPI and keep GitHub in sync. 