#!/usr/bin/env bash
set -euo pipefail

# One-time installer for EnvSense systemd services on Raspberry Pi.
# Usage:
#   bash scripts/setup_systemd.sh
# Optional env vars:
#   ENVSENSE_USER=engg1101
#   ENVSENSE_PROJECT_DIR=/home/engg1101/EnvSense-AI
#   ENVSENSE_VENV_PATH=/home/engg1101/EnvSense-AI/sensor_env
#   ENABLE_SENSOR_SERVICE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENVSENSE_USER="${ENVSENSE_USER:-$USER}"
ENVSENSE_PROJECT_DIR="${ENVSENSE_PROJECT_DIR:-$ROOT_DIR}"
ENVSENSE_VENV_PATH="${ENVSENSE_VENV_PATH:-${ENVSENSE_PROJECT_DIR}/sensor_env}"
ENABLE_SENSOR_SERVICE="${ENABLE_SENSOR_SERVICE:-true}"

if [[ ! -d "${ENVSENSE_PROJECT_DIR}" ]]; then
  echo "Project directory does not exist: ${ENVSENSE_PROJECT_DIR}"
  exit 1
fi

if [[ ! -x "${ENVSENSE_VENV_PATH}/bin/python" ]]; then
  echo "Virtualenv not found or invalid: ${ENVSENSE_VENV_PATH}"
  echo "Please create it first, e.g.:"
  echo "  python3 -m venv sensor_env"
  echo "  source sensor_env/bin/activate && pip install -r requirements.txt"
  exit 1
fi

render_template() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|__USER__|${ENVSENSE_USER}|g" \
    -e "s|__PROJECT_DIR__|${ENVSENSE_PROJECT_DIR}|g" \
    -e "s|__VENV_PATH__|${ENVSENSE_VENV_PATH}|g" \
    "${src}" > "${dst}"
}

tmp_web="$(mktemp)"
tmp_sensor="$(mktemp)"
cleanup() {
  rm -f "${tmp_web}" "${tmp_sensor}"
}
trap cleanup EXIT

render_template "${ROOT_DIR}/deploy/systemd/envsense-web.service.template" "${tmp_web}"
render_template "${ROOT_DIR}/deploy/systemd/envsense-sensor.service.template" "${tmp_sensor}"

echo "Installing systemd unit: envsense-web.service"
sudo cp "${tmp_web}" /etc/systemd/system/envsense-web.service

if [[ "${ENABLE_SENSOR_SERVICE}" == "true" ]]; then
  echo "Installing systemd unit: envsense-sensor.service"
  sudo cp "${tmp_sensor}" /etc/systemd/system/envsense-sensor.service
fi

sudo systemctl daemon-reload

echo "Enabling + restarting envsense-web.service"
sudo systemctl enable envsense-web.service
sudo systemctl restart envsense-web.service

if [[ "${ENABLE_SENSOR_SERVICE}" == "true" ]]; then
  echo "Enabling + restarting envsense-sensor.service"
  sudo systemctl enable envsense-sensor.service
  sudo systemctl restart envsense-sensor.service
fi

echo
echo "Service status:"
sudo systemctl --no-pager --full status envsense-web.service || true
if [[ "${ENABLE_SENSOR_SERVICE}" == "true" ]]; then
  sudo systemctl --no-pager --full status envsense-sensor.service || true
fi

echo
echo "Done. Quick commands:"
echo "  sudo systemctl restart envsense-web.service"
echo "  sudo systemctl status envsense-web.service"
echo "  sudo journalctl -u envsense-web.service -f"
if [[ "${ENABLE_SENSOR_SERVICE}" == "true" ]]; then
  echo "  sudo systemctl restart envsense-sensor.service"
  echo "  sudo systemctl status envsense-sensor.service"
  echo "  sudo journalctl -u envsense-sensor.service -f"
fi

