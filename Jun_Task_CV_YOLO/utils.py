# coding:cp1251
import cv2


def draw_detections(frame, detections):
    """
    Отрисовывает найденные на кадре объекты.

    :param frame: Кадр (numpy array).
    :param detections: Список детекций [(x1, y1, x2, y2, label, confidence), ..., (x_n, y_n, x_n, y_n, label, confidence)].
    :return: Изображение с отрисованными объектами.
    """
    for box, score in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"Person: {score:.2f}"

        # Рисуем bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return frame
