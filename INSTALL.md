# Installation Instructions

## Standard Dependencies

Install the standard dependencies using uv:

```bash
uv sync
```

## DGL Installation (Platform-Specific)

DGL (Deep Graph Library) requires platform-specific wheels and is NOT included in `pyproject.toml`.

### macOS (Development - CPU only)

**IMPORTANT:** As of June 2024, DGL stopped providing prebuilt packages for macOS. The last available version is 2.2.1.

**Option 1: Install from source (advanced)**
```bash
# Requires build tools and is complex - not recommended for development
git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
# ... follow build instructions
```

**Option 2: Use old version (2.2.1 - may have compatibility issues)**
```bash
uv run pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/repo.html
```

**Option 3: Develop on macOS, run on Linux (RECOMMENDED)**
The code is written to be platform-agnostic. You can develop and review code on macOS without DGL installed, then test/run on Linux. This is the cleanest workflow given DGL's current state on macOS.

### Linux with CUDA GPU (Production)

First, check your CUDA version:
```bash
nvcc --version
```

Then install the appropriate DGL wheel:

**For CUDA 11.8:**
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.9/cu118/repo.html
```

**For CUDA 12.1:**
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.9/cu121/repo.html
```

**For CUDA 12.4:**
```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.9/cu124/repo.html
```

### Verify Installation

```bash
uv run python -c "import dgl; print(f'DGL version: {dgl.__version__}')"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Complete Setup Script

**macOS:**
```bash
uv sync
uv pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

**Linux with GPU:**
```bash
uv sync
# Replace cu121 with your CUDA version
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.9/cu121/repo.html
```

## Notes

- The Python code is fully platform-agnostic
- Only DGL installation changes between platforms
- When moving code to a new machine, just re-run the appropriate installation commands
- Always use `uv run` to execute scripts to ensure you're using the isolated environment
