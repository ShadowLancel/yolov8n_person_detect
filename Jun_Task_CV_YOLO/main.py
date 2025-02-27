# coding:cp1251
import cv2
import torch
import numpy as np
from detector import PersonDetector
from utils import draw_detections
import time
import os

video_path = "crowd.mp4"
file_name = "crowd_detected.mp4"
output_path = "output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
detector = PersonDetector()

# Чтение видео
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Расширение выходного файла
new_width, new_height = 800, 600
out = cv2.VideoWriter(
    os.path.join(output_path, file_name), fourcc, fps, (new_width, new_height)
)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Уменьшение кадра, детекция людей и отрисовка bbox
    resized_frame = cv2.resize(frame, (new_width, new_height))
    detections = detector.detect(resized_frame)
    annotated_frame = draw_detections(resized_frame, detections)

    # Записываем кадр в выходной файл
    out.write(annotated_frame)

cap.release()
out.release()

cv2.destroyAllWindows()

end_time = time.time()

print(f"result time:{end_time - start_time} seconds")

if __name__ == "__main__":
    print("Processing completed. Video saved to", output_path)
