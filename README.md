# EnvSense AI

EnvSense AI is a personal environment assistant for Raspberry Pi. It reads your indoor air via:

- **DHT22** (temperature + humidity)
- **ENS160** (eCO2 + TVOC)
- Optional **SSD1306 (I2C OLED)** for local display

The system provides **one web dashboard** with **two scenario modules**:

1. **Room Reset Coach (Study Mode)**  
   Helps you decide whether the room is still suitable for studying daytime, and recommends what to do next.
2. **Sleep Guard (Sleep Mode)**  
   Helps you decide whether the current space is sleep-ready, or whether you should ventilate first.

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
python3 -m venv sensor_env
source sensor_env/bin/activate
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
- Optional: set `dht22.pin_candidates: [21, 4, 17]` to probe multiple BCM pins in order

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

Collect to a scenario-specific file when you want separate runs:

```bash
python3 main.py --csv-path data/realtime_study.csv
python3 main.py --csv-path data/realtime_sleep.csv
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

The front-end calls backend APIs every **10 seconds** in live mode (offline example mode uses a slower refresh).

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

Train with separate CSV files per scenario:

```bash
python3 models/train_models.py \
  --room-reset-csv data/realtime_study.csv \
  --sleep-csv data/realtime_sleep.csv
```

What it does:

- Trains **Room Reset Coach** model (`models/room_reset/model.joblib`)
- Trains **Sleep Guard** model (`models/sleep_guard/model.joblib`)
- Uses `data/realtime.csv` windows + optional discovered annotation tables
- Fits the models from recorded sensor windows, tracked study labels, and sleep feedback tables

Trained models are saved as `model.joblib`.

---

## Optional annotations

The training pipeline can auto-discover optional annotation tables anywhere under `data/`, or you can set explicit local overrides in `config_hardware/config.yaml` under `data_paths`.

### Study-mode annotations

Provide a CSV with:

```csv
timestamp,best_action
2026-03-19T10:20:00Z,Open window
2026-03-19T10:45:00Z,Stay
```

Accepted action labels:

- `Stay`
- `Open window`
- `Open door`
- `Move soon`

### Sleep-mode feedback

Provide a CSV with a `timestamp` column and one categorical feedback column, for example:

```csv
timestamp,feedback
2026-03-20T00:00:00Z,slept_well
```

Accepted feedback values:

- `slept_well`
- `okay`
- `poor_sleep`

The trainer maps these to:

- `Good to sleep`
- `Sleep okay after ventilating`
- `Not ideal yet`

### Practical workflow

1. Run the logger to collect `data/realtime.csv`.
2. Save or copy the scenario CSV you want to train on.
3. Place any annotation CSVs under `data/` in your preferred folder structure, or point `data_paths` to them explicitly.
4. Keep `timestamp` aligned with the collected series; optional context columns are allowed and ignored by training.

---

## Notes for deployment

Typical workflow:

1. Start the sensor logger on the Pi (`python3 main.py`)
2. Start the web app (`uvicorn fastapi_app:app --host 0.0.0.0 --port 8501`)
3. Train/update models as you collect data (`python3 models/train_models.py`)

---

## Run as systemd services (recommended on Raspberry Pi)

This project includes service templates and an installer script so the app starts on boot.

### One-time setup

From repo root on Pi:

```bash
cd ~/EnvSense-AI
bash scripts/setup_systemd.sh
```

If your username/project path/venv path differ, pass overrides:

```bash
ENVSENSE_USER=engg1101 \
ENVSENSE_PROJECT_DIR=/home/engg1101/EnvSense-AI \
ENVSENSE_VENV_PATH=/home/engg1101/EnvSense-AI/sensor_env \
bash scripts/setup_systemd.sh
```

### What this installs

- `envsense-web.service` -> runs `uvicorn fastapi_app:app --host 0.0.0.0 --port 8501`
- `envsense-sensor.service` -> runs `python main.py`

### Service control

```bash
sudo systemctl status envsense-web.service
sudo systemctl restart envsense-web.service
sudo journalctl -u envsense-web.service -f
```

```bash
sudo systemctl status envsense-sensor.service
sudo systemctl restart envsense-sensor.service
sudo journalctl -u envsense-sensor.service -f
```

