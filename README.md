# ChilliGuard

Food adulteration detection system for red chilli powder. The project includes a Flask REST API and a Vite + React frontend.

## Required Model Files

Place these trained artifacts in this project folder:

- `best_model.h5`
- `class_names.json`
- `label_map.json`
- `model_stats.json`

Optional visual outputs served by `/static/<filename>`:

- `confusion_matrix.png`
- `training_curves.png`
- `sample_predictions.png`

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Flask API:

```bash
python app.py
```

The API starts at:

```text
https://chilligard-backend.onrender.com
```

Start the React frontend:

```bash
npm install
npm run dev
```

The website starts at:

```text
http://localhost:5173
```

## Endpoints

### Health

```bash
curl https://chilligard-backend.onrender.com/health
```

Returns:

```json
{"status": "ok", "model_loaded": true}
```

### Predict

Upload an image using multipart form-data with field name `image`.

```bash
curl -X POST https://chilligard-backend.onrender.com/predict \
  -F "image=@sample.jpg"
```

### Model Info

```bash
curl https://chilligard-backend.onrender.com/model-info
```

Returns the full contents of `model_stats.json`.

### Static Visualizations

```text
https://chilligard-backend.onrender.com/static/confusion_matrix.png
https://chilligard-backend.onrender.com/static/training_curves.png
https://chilligard-backend.onrender.com/static/sample_predictions.png
```
