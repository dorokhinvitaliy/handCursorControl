import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

click_threshold = 60  # Порог для левого клика
three_fingers_threshold = 80  # Порог для сведения трёх пальцев
click_cooldown = 1  # секунд между кликами

last_click_time = 0

prev_time = 0 # for cheching FPS
errors = []

def get_landmark_px(landmark, frame_width, frame_height):
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            index_x, index_y = get_landmark_px(index_finger_tip, screen_width, screen_height)

            error = math.hypot(index_x - pyautogui.position().x, index_y - pyautogui.position().y)
            errors.append(error)


            # Расстояния между пальцами
            dist_thumb_index = math.hypot(index_x - int(thumb_tip.x * screen_width), index_y - int(thumb_tip.y * screen_height))
            dist_thumb_middle = math.hypot(int(middle_finger_tip.x * screen_width) - int(thumb_tip.x * screen_width),
                                           int(middle_finger_tip.y * screen_height) - int(thumb_tip.y * screen_height))
            dist_index_middle = math.hypot(index_x - int(middle_finger_tip.x * screen_width),
                                           index_y - int(middle_finger_tip.y * screen_height))

            current_time = time.time()

            # Обработка правого клика (три пальца сведены)
            if (dist_thumb_index < three_fingers_threshold and
                dist_thumb_middle < three_fingers_threshold and
                dist_index_middle < three_fingers_threshold):
                if (current_time - last_click_time) > click_cooldown:
                    pyautogui.rightClick()

                    click_time = time.time()
                    print("Клик:", click_time)
                    print("Задержка:", (click_time - gesture_time) * 1000, "мс")

                    last_click_time = current_time
            # Левый клик (одиночное сведение большого и указательного)
            elif dist_thumb_index < click_threshold:
                if (current_time - last_click_time) > click_cooldown:
                    pyautogui.click()

                    click_time = time.time()
                    print("Клик:", click_time)
                    print("Задержка:", (click_time - gesture_time) * 1000, "мс")

                    last_click_time = current_time
            else:
                pyautogui.moveTo(index_x, index_y)

            # Визуализация
            ix, iy = get_landmark_px(index_finger_tip, w, h)
            tx, ty = get_landmark_px(thumb_tip, w, h)
            mx, my = get_landmark_px(middle_finger_tip, w, h)
            rx, ry = get_landmark_px(ring_finger_tip, w, h)

            cv2.circle(frame, (ix, iy), 10, (255, 0, 255), -1)
            cv2.circle(frame, (tx, ty), 10, (255, 0, 255), -1)
            cv2.circle(frame, (mx, my), 10, (255, 0, 255), -1)
            cv2.circle(frame, (rx, ry), 10, (255, 0, 255), -1)
            cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
            cv2.line(frame, (tx, ty), (mx, my), (0, 255, 0), 2)
            cv2.line(frame, (ix, iy), (mx, my), (0, 255, 0), 2)
            cv2.line(frame, (ix, iy), (rx, ry), (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if errors: print("Средняя ошибка: ", sum(errors)/len(errors))

cap.release()
cv2.destroyAllWindows()
