# 🚗 CarValidation-System

Система для автоматического определения **царапин, вмятин и грязи на автомобилях** по изображению, с помощью модели YOLOv8.

---

## ⚙️ Установка проекта

Склонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/abo-code-1/CarValidation-System.git
cd CarValidation-System

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Для macOS/Linux
# .venv\Scripts\activate   # Для Windows

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
