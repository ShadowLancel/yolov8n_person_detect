# coding:cp1251
import torch
from ultralytics import YOLO
import numpy as np


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", device=None):
        """
        Инициализация детектора.

        :param model_path: путь к предобученной модели YOLO. При отсутствии модели по данному пути сделает её установку из интернета.
        :param conf_threshold: порог уверенности для фильтрации объектов.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame, conf_threshold=0.5):
        """
        Выполняет детекцию людей на кадре.

        :param frame: входной кадр (numpy array).
        :return: список обнаруженных объектов в формате (x1, y1, x2, y2, label, confidence).
        """
        results = self.model(frame)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                # Проверяем, что класс == человек (0) и оценка модели не меньше порога
                if int(cls) == 0 and score >= conf_threshold:
                    detections.append((box, score))
        return detections
