# Lending Club Underwriting Lab

An English-language risk dashboard and client-side underwriting research simulator built from the Lending Club files in the parent directory.

## Run locally

```bash
npm install
npm run dev
```

## Rebuild model artifacts

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/train_model.py --data-dir ..
```

The training process writes aggregate dashboard data and deployable model artifacts under `public/`. Raw CSV files remain outside the website project and are never included in the build.

## Important limitation

This project is for research and demonstration only. It is not validated for real credit decisions or adverse-action notices.
