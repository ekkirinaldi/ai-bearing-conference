# AGENTS.md

## Cursor Cloud specific instructions

This is a Python data science project for bearing fault detection using Explainable AI (SHAP). There are no web services, databases, or Docker containers — only Jupyter notebooks and a `requirements.txt`.

### Project structure

- **Notebooks** live under `kaggle/input/cwru-bearing-datasets/` (CWRU) and `kaggle/input/ieee-phm-2012/` (IEEE PHM).
- **Primary dataset** (`feature_time_48k_2048_load_1.csv`) is checked into git under `kaggle/input/cwru-bearing-datasets/`.
- The IEEE PHM 2012 notebook requires a separate dataset not included in the repo.

### Running

- Activate venv: `source /workspace/.venv/bin/activate`
- Start Jupyter: `jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token='' --ServerApp.password=''`
- Notebooks use CPU fallback automatically when CUDA/MPS are unavailable.

### Linting

- Use `nbqa flake8 <notebook.ipynb>` for linting notebooks (installed in venv alongside `flake8`).
- Style warnings (E231, E261, etc.) are typical for data-science notebooks and non-blocking.

### Testing

- No automated test suite exists. Validate by running notebook cells or executing equivalent Python scripts.
- The main ML pipeline (SVC on CWRU data) runs in ~1 second and achieves ~95% accuracy — use it as a quick smoke test.

### Gotchas

- `python3.12-venv` system package must be installed before creating the venv (`sudo apt-get install -y python3.12-venv`).
- PyTorch downloads CUDA/NVIDIA wheels (~2 GB); installs take 1-2 minutes even on fast connections.
