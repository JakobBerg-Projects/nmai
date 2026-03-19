import cv2
from ultralytics import YOLO


def detect(source: str, model_path: str = "yolov8n.pt", conf: float = 0.25):
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf)

    for result in results:
        img = result.orig_img.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            score = float(box.conf[0])
            label = f"{result.names[cls]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Detections", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    detect(source="data/sample.jpg")
