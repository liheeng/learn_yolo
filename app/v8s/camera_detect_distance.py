from ultralytics import YOLO
import cv2

# model = YOLO("best.pt")  # 你的掼蛋/检测模型
# 1. 加载掼蛋项目专用 YOLOv8s 模型
model = YOLO("lib/yolov8s.pt")
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率，适配识别精度与画面流畅度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 测距标定参数（新手直接用，后期微调即可）
KNOWN_WIDTH = 8.0  # 真实物体宽度（卡牌约8cm）
FOCAL_LENGTH = 520  # 相机焦距固定值


def get_distance(px_width):
    """单目测距公式：距离(cm) = (真实宽 * 焦距) / 像素宽"""
    if px_width == 0:
        return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / px_width


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640, conf=0.5)
    frame = results[0].plot()

    # 遍历每个物体，计算距离
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        px_w = x2 - x1
        dist = get_distance(float(px_w))

        # 在画面显示距离
        cv2.putText(
            frame,
            f"{dist:.1f}cm",
            (int(x1), int(y1) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),  # 红色
            2,
        )

        print(f"检测物体距离：{dist:.1f} cm")

    cv2.imshow("YOLO Distance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
