#!/bin/bash
set -e

# =================================================================
# TU WPISZ LINKI DO TWOICH MODELI .HEF (Z URL lub lokalnego serwera)
# =================================================================
URL_DETECTOR="https://github.com/winek87/Modele/releases/download/0.1/yolov8n_relu6_face_kpts--640x640_quant_hailort_hailo8_1.hef"
URL_RECOGNIZER="https://github.com/winek87/Modele/releases/download/0.1/Buffalo_L.hef"

# Nazwy plików oczekiwane przez skrypt
DETECTOR_FILE="yolov8n_relu6_face_kpts--640x640_quant_hailort_hailo8_1.hef"
RECOGNIZER_FILE="Buffalo_L.hef"

# Pobieranie modeli tylko jeśli ich nie ma
if [ ! -f "$DETECTOR_FILE" ]; then
    echo "⬇️ Pobieranie modelu detekcji..."
    wget -q -O "$DETECTOR_FILE" "$URL_DETECTOR" || echo "❌ Błąd pobierania detektora. Sprawdź URL!"
else
    echo "✅ Model detekcji już jest."
fi

if [ ! -f "$RECOGNIZER_FILE" ]; then
    echo "⬇️ Pobieranie modelu rozpoznawania..."
    wget -q -O "$RECOGNIZER_FILE" "$URL_RECOGNIZER" || echo "❌ Błąd pobierania recognizera. Sprawdź URL!"
fi

# Ustawienie zmiennych dla server.py
export DETECTOR_HEF="$DETECTOR_FILE"
export RECOGNIZER_HEF="$RECOGNIZER_FILE"

echo "🚀 Uruchamianie serwera..."
exec python3 server.py
