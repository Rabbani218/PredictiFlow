# 🔮 PredictiFlow

A professional time series analysis and forecasting platform that democratizes data science for everyone.

## 🌟 Features

- **Automated Time Series Analysis**: Upload your CSV and get instant insights
- **Smart Model Selection**: Automatically selects the best forecasting model
- **Interactive Visualizations**: Beautiful, interactive charts powered by Plotly
- **Easy to Use**: No data science expertise required
- **Production Ready**: Built with scalability and reliability in mind

## 🏗️ System Architecture

```
[Frontend (Streamlit)] -> [Backend API (FastAPI)] -> [ML Engine (Prophet, ARIMA)]
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML/Data Processing**: 
  - pandas
  - Prophet
  - statsmodels
  - scikit-learn
- **Visualization**: Plotly
- **Deployment**: Docker & Docker Compose

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Git

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/predictiflow.git
# PredictiFlow 🔮

PredictiFlow is a lightweight, reproducible time-series forecasting playground.
It provides:

- A FastAPI backend that runs forecasting pipelines (Prophet / SARIMA / ETS fallback)
- A Streamlit frontend for uploading CSVs, running forecasts (sync or async), and visualizing results
- Optional Redis + RQ worker for background jobs
- Notebooks with data exploration and API integration examples

This repository is intended as a developer-friendly demo and a starting point for production hardening.

## Quickstart — Recommended (Docker)

The easiest way to run everything (backend, frontend, Redis worker) is with Docker.

1. Install Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Linux).
2. From the project root run:

```powershell
docker compose up --build
```

The frontend (Streamlit) will be available at http://localhost:8501 and the backend at http://localhost:8000.

## Quickstart — Local (without Docker)

1. Create a short virtualenv and install requirements. We provide an orchestration helper `Autorun.py` to automate common steps.

```powershell
2. Build and run with Docker Compose:
```bash
docker-compose up --build

If you prefer to run steps manually:

```powershell
```

3. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs


Notes:
- On Windows, if `pip install` fails for heavy packages (numpy, scipy, etc.), consider using Docker or a conda environment to pick up prebuilt binary wheels.
- `Autorun.py` will try simple self-healing (upgrade pip, retry using `--only-binary`), and logs operations to `logs/autorun.log`.

## Using the async job queue

If you run via Docker Compose, a Redis service and RQ worker are included in the compose stack. Submit async jobs from the UI and monitor `/jobs` endpoints.

## Developer notes

- Tests: `pytest -q` runs the small test suite.
- Core backend module: `backend/app/core/forecasting.py`
- API entrypoint: `backend/app/main.py`
- Streamlit UI: `frontend/app.py`

## Troubleshooting

- If you see Streamlit "Missing ScriptRunContext" when running `python frontend/app.py`, run it via `streamlit run frontend/app.py` (or use `python Autorun.py all` which will handle this).
- If `pip install` fails with build errors on Windows for packages like `numpy`, use Docker or create a conda environment: `conda create -n pf python=3.10 pandas numpy scikit-learn` then `conda activate pf`.

## Contributing

Contributions are welcome — open an issue with a short description of the feature or bug.

## License

MIT

## 📂 Project Structure

```
predictiflow/
├── backend/           # FastAPI application
├── frontend/         # Streamlit web interface
├── notebooks/        # Jupyter notebooks for research
├── data/            # Sample datasets
└── docker/          # Docker configuration
```

## 📊 Sample Data

Sample time series datasets are provided in the `data/` directory for testing.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Run locally (without Docker)

If you prefer to run the services locally on Windows (PowerShell), create and activate a virtual environment and install backend dependencies:

```powershell
cd 'c:\Users\DELL\OneDrive\Documents\Project Github\PredictiFlow'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel build
pip install -r backend/requirements.txt
```

Start the backend and frontend in separate terminals (you can use `scripts/start-dev.ps1`):

Backend:
```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:
```powershell
cd frontend
streamlit run app.py
```

## Troubleshooting (Windows / Python 3.13)

- Building some packages (like numpy/pandas) from source on Windows can fail if the build toolchain is missing. Two recommended approaches:
  1. Use Docker (preferred) — the project Dockerfiles include build tools so images can compile wheels inside the container.
  2. Use a supported Python version like 3.10 or 3.11 in a virtual environment where pre-built wheels are available.

- If you encounter errors during `pip install -r backend/requirements.txt`, try installing `build`, `setuptools`, and `wheel` first:

```powershell
python -m pip install --upgrade pip setuptools wheel build
```

Then retry installing requirements.

### Windows long path errors and a quick workaround

- Problem: On Windows you may see errors like `OSError: [Errno 2] No such file or directory ... jupyterlab-manager ...` when installing packages. This is frequently caused by Windows long-path limits.
- Short-term workaround (no admin required): create a virtualenv in a short path (e.g., `C:\pf_venv`) and install dev packages there:

```powershell
# 🔮 PredictiFlow

PredictiFlow is a developer-friendly time-series forecasting playground and demo platform.
It provides a FastAPI backend that runs an automated forecasting pipeline (with lightweight fallbacks),
a Streamlit frontend for uploads and visualization, and a small model registry for storing artifacts.

This README targets Windows PowerShell users by default but includes Docker and Conda instructions for cross-platform reproducibility.

## Highlights

- Upload CSVs and get automated EDA + forecasting
- Smart model fallbacks (Prophet / ARIMA / Exponential smoothing or simple baselines when heavy libs are missing)
- Multi-series helpers and demo datasets (hypercar, gold, harga_laptop)
- Lightweight model registry (SQLite + files) with REST endpoints
- Notebooks demonstrating examples and guided demos

---

## Quickstart — Recommended (Docker)

Docker gives the most reproducible experience (prebuilt wheels and required services).

1. Install Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Linux).
2. From the project root run in PowerShell:

```powershell
docker compose up --build
```

Services:
- Backend API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

To stop and remove containers:

```powershell
docker compose down --volumes --remove-orphans
```

---

## Quickstart — Local (PowerShell)

If you prefer not to use Docker, a Conda environment is recommended on Windows to avoid building heavy scientific packages.

Conda (recommended):

```powershell
# create and activate conda env (requires conda/Miniconda/Anaconda installed)
conda env create -f environment.yml
conda activate predictiflow
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
```

Virtualenv (short path recommended if you hit Windows long-path issues):

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel build
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

Run backend and frontend in separate shells:

Backend (from repo root):

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (Streamlit):

```powershell
Set-Location frontend
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or use the helper:

```powershell
.\scripts\start-dev.ps1
```

---

## Running the tests

Run the unit tests with pytest from the project root (activate your `.venv` or conda env first):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Notes:
- Tests are written to avoid touching the repository registry DB by default — they use temporary paths during execution.
- The core forecasting pipeline safely falls back when seasonal decomposition can't be performed on short series.

---

## Notebooks and demos

Notebooks live in `notebooks/` and demonstrate EDA, model selection, registry usage, and multi-series samples.

- `notebooks/04-Multi-Series-Samples.ipynb` — multi-series demo using `data/hypercar_sales.csv` and `data/gold_sales.csv`.
- `notebooks/07-Process-Harga-Laptop.ipynb` — walkthrough for `data/harga_laptop.csv` (real-world messy CSV).

Lightweight demos (scripts):

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/demo_hypercar.py
python scripts/run_harga_laptop.py
```

---

## Data

Sample datasets live in `data/`:

- `harga_laptop.csv` — real-world laptop pricing dataset (messy formatting, varied encodings)
- `gold_sales.csv` — sample multi-product time series
- `hypercar_sales.csv` — toy multi-series dataset grouped by `model` and `brand`

The pipeline includes helpers to detect date and value columns and to parse localized numeric formats; however, some CSVs may need encoding fallbacks (the runner script already attempts a latin-1 fallback).

---

## API overview

Key endpoints (FastAPI):

- GET /health — health check
- POST /predict — synchronous CSV forecast (multipart/form-data; file field `file`, optional `periods` query param)
- POST /predict-async — enqueue a background forecast job (uses BackgroundTasks or RQ if Redis configured)
- GET /jobs — list recent jobs
- GET /jobs/{id} — get job status/result
- GET /models — list registered models
- POST /models — upload + register a model (multipart file)
- GET /models/{id}/download — download model artifact file

Example cURL (sync predict):

```powershell
curl -F "file=@data/sample_sales.csv" -F "periods=30" http://localhost:8000/predict
```

---

## Model registry

Registry is a lightweight SQLite DB and a `models/` folder under the `backend` folder by default.

You can override registry locations with environment variables (useful for tests or CI):

- `PREDICTIFLOW_REGISTRY_DB` — path to the SQLite DB file
- `PREDICTIFLOW_MODELS_DIR` — directory where model files are stored

Example (PowerShell):

```powershell
$env:PREDICTIFLOW_REGISTRY_DB = "C:\tmp\pf_registry.db"
$env:PREDICTIFLOW_MODELS_DIR = "C:\tmp\pf_models"
```

---

## Troubleshooting

- If `pip install` fails for heavy packages on Windows, prefer the Conda route or use Docker.
- If the Streamlit app shows "Missing ScriptRunContext", start it with `streamlit run frontend/app.py`.
- If forecasting fails with errors about seasonal decomposition on very short series, ensure your CSV has at least ~60 rows for monthly seasonality; otherwise the code will now safely skip decomposition and continue with fallback forecasts.

If you hit anything that looks like an unexpected exception, please open an issue with the failing traceback and the CSV sample that reproduces it.

---

## Development notes

- Core backend: `backend/app/core/forecasting.py`
- API entrypoint: `backend/app/main.py`
- Registry: `backend/app/registry.py`
- Frontend: `frontend/app.py` (Streamlit)

Follow the repository style and add tests when changing behavior that affects API outputs or registry semantics.

---

## Contributing

Contributions are welcome. Please open issues or PRs and include tests for non-trivial changes.

---

## License

MIT — see the `LICENSE` file.
MIT — lihat file `LICENSE`.

---

Terima kasih telah menggunakan PredictiFlow. Jika Anda ingin bantuan tambahan (tayangkan demo, terapkan CI, atau terjemahkan catatan teknis), beri tahu saya tugas berikutnya.
