# Управление курсора с помощью жестов руки

## Описание
Данная программа использует веб-камеру для отслеживания руки и перемещения курсора в зависимости от положения кончика указательного пальца. Основана на OpenCV, Mediapipe и PyAutoGUI.

## Требования
- Python 3.7+
- Установленные библиотеки:
  - OpenCV
  - Mediapipe
  - PyAutoGUI

## Установка

1. **Установите Python** (если не установлен):
   - [Скачать Python](https://www.python.org/downloads/)

2. **Склонируйте репозиторий или скачайте код:**
   ```bash
   git clone https://git.miem.hse.ru/blank8190307/handCursorControl.git
   cd handCursorControl
   ```

3. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для macOS/Linux
   venv\Scripts\activate  # Для Windows
   ```

4. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```
   **ИЛИ**
   ```bash
   pip install opencv-python mediapipe pyautogui
   ```

## Запуск программы

```bash
python handCursorControl.py
```

## Управление
- Двигайте указательный палец перед камерой, и курсор будет следовать за ним.
- Чтобы выйти, нажмите `q` в окне отображения видео.

## Возможные ошибки и их решения
1. **Ошибка `ModuleNotFoundError: No module named 'mediapipe'`**
   - Убедитесь, что библиотека установлена:
     ```bash
     pip install mediapipe
     ```

2. **Ошибка `cv2.VideoCapture(0) not working`**
   - Убедитесь, что веб-камера не используется другими приложениями.
   - Проверьте доступ к камере в настройках системы.

