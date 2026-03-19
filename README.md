# EnvSense AI

EnvSense AI is a personal environment assistant for Raspberry Pi. It reads your indoor air via:

- **DHT22** (temperature + humidity)
- **ENS160** (eCO2 + TVOC)
- Optional **SSD1306 (I2C OLED)** for local display

The system provides **one Streamlit website** with **two scenario modules**:

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

Debug (finite loop):

```bash
python3 main.py --iterations 100
```

CSV columns:

- `timestamp` (ISO 8601)
- `temp_C`, `humidity`, `eco2_ppm`, `tvoc`

---

## Run the Streamlit website

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

The UI auto-refreshes every **10 seconds** using `streamlit-autorefresh`.

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
python3 train_models.py
```

What it does:

- Trains **Room Reset Coach** model (`models/room_reset/model.joblib`)
- Trains **Dorm Sleep Guard** model (`models/sleep_guard/model.joblib`)
- If realtime training data is missing/insufficient, it trains on **synthetic bootstrap data**

Trained models are saved as `model.joblib`.

---

## Session data organization

The repo contains folders for future session-based labeling/feedback:

- `data/room_reset_sessions/`
- `data/sleep_sessions/`

Currently, the app includes a placeholder morning check-in block for Sleep Mode, and the code includes hooks for reading optional label CSV files if you add them later.

---

## Notes for real deployment

Typical workflow:

1. Start the sensor logger on the Pi (`python3 main.py`)
2. Start the Streamlit app (`streamlit run dashboard/app.py ...`)
3. Train/update models as you collect data (`python3 train_models.py`)

