# Używamy sprawdzonej wersji Python 3.13
FROM python:3.13-slim

# Zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Instalacja podstawowych narzędzi i zależności dla OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalacja zależności Python
RUN pip install --upgrade pip
RUN pip install flask opencv-python-headless numpy requests pillow pillow-heif rawpy

# 3. Pobieranie i instalacja HAILO (Metoda Pancerna)
WORKDIR /tmp_install

# Pobieranie plików
RUN wget -q http://dev-public.hailo.ai/2025_10/Hailo8/hailort_4.23.0_arm64.deb
RUN wget -q http://dev-public.hailo.ai/2025_10/Hailo8/hailo-tappas-core_5.1.0_arm64.deb
RUN wget -q http://dev-public.hailo.ai/2025_10/Hailo8/hailo_tappas_core_python_binding-5.1.0-py3-none-any.whl
RUN wget -q http://dev-public.hailo.ai/2025_10/Hailo8/hailort-4.23.0-cp313-cp313-linux_aarch64.whl

# === FIX TOTALNY ===
# Krok A: Tworzymy fałszywy systemctl. 
# Dzięki temu skrypt instalacyjny Hailo nie wywali błędu "command not found".
RUN echo '#!/bin/sh\nexit 0' > /usr/bin/systemctl && chmod +x /usr/bin/systemctl

# Krok B: Instalujemy pakiety przez APT (a nie dpkg).
# Podanie ścieżki "./plik.deb" sprawia, że apt instaluje ten plik LUBIANE dociągając zależności!
RUN apt-get update
RUN apt-get install -y ./hailo-tappas-core_5.1.0_arm64.deb || true
RUN apt-get install -y ./hailort_4.23.0_arm64.deb || true

# Krok C: Usuwamy fałszywkę
RUN rm /usr/bin/systemctl
# ===================

# Instalacja Python Bindings (.whl)
RUN pip install hailort-4.23.0-cp313-cp313-linux_aarch64.whl
RUN pip install hailo_tappas_core_python_binding-5.1.0-py3-none-any.whl

# Sprzątanie
WORKDIR /app
RUN rm -rf /tmp_install

# 4. Kopiowanie aplikacji
COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 5. Start
EXPOSE 8000
CMD ["./entrypoint.sh"]
