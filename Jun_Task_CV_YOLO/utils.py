# coding:cp1251
import cv2


def draw_detections(frame, detections):
    """
    ������������ ��������� �� ����� �������.

    :param frame: ���� (numpy array).
    :param detections: ������ �������� [(x1, y1, x2, y2, label, confidence), ..., (x_n, y_n, x_n, y_n, label, confidence)].
    :return: ����������� � ������������� ���������.
    """
    for box, score in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"Person: {score:.2f}"

        # ������ bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return frame
