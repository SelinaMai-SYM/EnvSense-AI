# EnvSense AI

EnvSense AI is a personal environment assistant for Raspberry Pi. It reads your indoor air via:

- **DHT22** (temperature + humidity)
- **ENS160** (eCO2 + TVOC)
- Optional **SSD1306 (I2C OLED)** for local display

The system provides **one web dashboard** with **two scenario modules**:

1. **Room Reset Coach (Study Mode)**  
   Helps you decide whether the room is still suitable for studying daytime, and recommends what to do next.
2. **Dorm Sleep Guard (Sleep Mode)**  
   Helps you decide whether the dorm room is sleep-ready, or whether you should ventilate first.

Both modules share the same **sensing pipeline** and use **lightweight ML (RandomForest)** with safe baseline fallbacks when models are not available.

---

## Hardware

- Raspberry Pi running **Raspberry Pi OS** (Python 3)
- **DHT22** for `temp_C`, `humidity`
- **ENS160** for `eco2_ppm`, `tvoc`
- Optional **SSD1306** OLED display

---

## Install dependencies

Recommended on Raspberry Pi (virtual environment first):

```bash
cd "EnvSense-AI"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

System packages commonly needed for hardware libraries:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev libgpiod2 i2c-tools
```

---

## Configure hardware

Edit `config_hardware/config.yaml`:

- Ensure wiring and I2C are enabled on Raspberry Pi
- DHT22 pin uses BCM numbering (default `21`, same as workshop example style)

If `config.yaml` is missing, the project falls back to `config_hardware/config.example.yaml`.

Enable I2C once on Pi:

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
```

Check I2C devices:

```bash
i2cdetect -y 1
```

---

## Run the sensor logger

This continuously reads sensors every `sample_interval_sec` and appends one CSV row into `data/realtime.csv`.

```bash
python3 main.py
```

Collect to a location-specific file (recommended for cleaner datasets):

```bash
python3 main.py --csv-path data/realtime_dorm_room.csv
python3 main.py --csv-path data/realtime_library.csv
```

Current project setup (two locations):

```bash
python3 main.py --csv-path data/realtime_classroom.csv
python3 main.py --csv-path data/realtime_bedroom.csv
```

Debug (finite loop):

```bash
python3 main.py --iterations 100
```

CSV columns:

- `timestamp` (ISO 8601)
- `temp_C`, `humidity`, `eco2_ppm`, `tvoc`

---

## Run the Web Dashboard (FastAPI + HTML/CSS/JS)

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8501 --reload
```

The front-end calls backend APIs every **10 seconds** in Realtime mode (offline demo uses a slower refresh).

---

## How AI / ML is incorporated

Each module uses:

- **Interpretable baseline rules** (always available)
- A **lightweight RandomForestClassifier** (joblib model trained locally)

Inference on the Pi:

- If `models/**/model.joblib` exists, the model is used
- If a model is missing, the system falls back to the baseline rules

---

## Train models locally

Training is intended for your laptop/local Python environment:

```bash
python3 models/train_models.py
```

Train with separate CSV files per model/location:

```bash
python3 models/train_models.py \
  --room-reset-csv data/realtime_classroom.csv \
  --sleep-csv data/realtime_bedroom.csv
```

What it does:

- Trains **Room Reset Coach** model (`models/room_reset/model.joblib`)
- Trains **Dorm Sleep Guard** model (`models/sleep_guard/model.joblib`)
- Uses `data/realtime.csv` windows + optional human labels from session folders
- If data is missing/insufficient, it backfills with **synthetic bootstrap data**

Trained models are saved as `model.joblib`.

Synthetic-only training (debug):

```bash
python3 models/train_models.py --force-synthetic
```

---

## Session data organization

The repo supports optional manual labels in session folders:

- `data/room_reset_sessions/`
- `data/sleep_sessions/`

### Room Reset label format

Create a session folder, e.g. `data/room_reset_sessions/2026-03-19-evening/`, then add `labels.csv` (or `actions.csv`):

```csv
timestamp,best_action
2026-03-19T10:20:00Z,Open window
2026-03-19T10:45:00Z,Stay
```

`best_action` must be one of:

- `Stay`
- `Open window`
- `Open door`
- `Move soon`

### Sleep Guard morning feedback format

Create a session folder, e.g. `data/sleep_sessions/2026-03-19-night/`, then add `morning_feedback.csv`:

```csv
timestamp,morning_feedback
2026-03-20T00:00:00Z,slept_well
```

`morning_feedback` must be one of:

- `slept_well`
- `okay`
- `poor_sleep`

The trainer maps these to:

- `Good to sleep`
- `Sleep okay after ventilating`
- `Not ideal yet`

### Practical labeling workflow

1. Run logger on Raspberry Pi to collect `data/realtime.csv`.
2. Copy the CSV to local machine for training.
3. Create a new session folder (do not overwrite previous sessions), for example:
   - `data/room_reset_sessions/2026-03-21-library/labels.csv`
   - `data/sleep_sessions/2026-03-21-dorm/morning_feedback.csv`
4. Edit only the required columns:
   - Room Reset: `timestamp`, `best_action`
   - Sleep Guard: `timestamp`, `morning_feedback`
5. Optional metadata columns (e.g. `location`, `environment_note`) are allowed and ignored by training, but useful for your records.

To avoid mixing different places/environments, keep them in separate session folders and use descriptive session names.

---

## Notes for real deployment

Typical workflow:

1. Start the sensor logger on the Pi (`python3 main.py`)
2. Start the Streamlit app (`streamlit run dashboard/app.py ...`)
3. Train/update models as you collect data (`python3 models/train_models.py`)

