from ultralytics import YOLO
import cv2

# 1. 加载掼蛋项目专用 YOLOv8s 模型
model = YOLO("lib/yolov8s.pt")

# 2. 打开本地摄像头（0为默认摄像头，USB外接摄像头可换1、2）
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率，适配识别精度与画面流畅度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("===== 掼蛋卡牌实时识别启动成功 =====")
print("操作说明：按 ESC 键退出画面")

# 3. 实时循环检测画面
while cap.isOpened():
    # 读取摄像头画面
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败，退出程序")
        break

    # YOLOv8s推理检测，固定640分辨率（掼蛋项目标准）
    results = model(frame, imgsz=640, conf=0.5)

    # 绘制检测框、类别、置信度到画面上
    annotated_frame = results[0].plot()

    # ========== 核心：输出识别结果（对接你的掼蛋AI模型）==========
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        # 识别类别、置信度、卡牌坐标
        cls_list = boxes.cls.cpu().numpy().tolist()
        conf_list = boxes.conf.cpu().numpy().tolist()
        pos_list = boxes.xyxy.cpu().numpy().tolist()
        print(f"当前识别卡牌数：{len(cls_list)}")
        print(f"卡牌类别：{cls_list}")
        print(f"卡牌坐标：{pos_list}\n")

    # 弹窗显示实时画面+检测标注
    cv2.imshow("Guandan YOLOv8s Camera Detect", annotated_frame)

    # 按ESC键退出窗口（固定按键）
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放资源、关闭窗口
cap.release()
cv2.destroyAllWindows()
