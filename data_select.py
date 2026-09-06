from ultralytics import YOLO
import os
import random

# 1. 训练好的权重
WEIGHTS_PATH = r"runs/xx"

# 2. 验证集图片文件夹
IMAGE_SOURCE = r"datasets/xx"

# 3. 可视化结果保存目录
OUTPUT_SAVE_DIR = r"runs/xx"

# 4. 固定配置：所有框粗细
BOX_THICKNESS = 2         

# 固定颜色表：保证同类目标永远同色
# 你可以自己改 RGB 颜色，(B, G, R) 格式
FIXED_COLORS = {
    0: (0, 255, 0),      # 类别0：绿色
    1: (255, 0, 0),      # 类别1：蓝色
    2: (0, 0, 255),      # 类别2：红色
    3: (255, 255, 0),    # 类别3：青色
    4: (255, 0, 255),    # 类别4：紫色
    5: (0, 255, 255),    # 类别5：黄色
    6: (128, 0, 128),    # 类别6：深紫
    7: (255, 165, 0),    # 类别7：橙色
    8: (0, 128, 0),      # 类别8：深绿
    9: (128, 128, 0),    # 类别9：橄榄色
}

def get_fixed_color(cls_id):
    """根据类别ID返回固定颜色，没有则自动生成稳定颜色"""
    if cls_id in FIXED_COLORS:
        return FIXED_COLORS[cls_id]
    # 自动生成固定随机色（同类别永远同色）
    random.seed(cls_id)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def yolo11_predict_and_visualize(weights, source, save_dir):
    # 加载模型
    model = RTDETR(weights)

    print("=" * 50)
    print("🚀 YOLOv11 推理可视化开始（仅显示边界框，固定类别颜色）")
    print(f"权重：{weights}")
    print(f"图片：{source}")
    print("=" * 50)

    # 推理
    results = model.predict(
        source=source,
        save=False,        # 关闭自动保存，手动绘制
        project=save_dir,
        name="result",
        exist_ok=True,
        conf=0.25,
        verbose=False
    )

    # 创建保存文件夹
    save_img_dir = os.path.join(save_dir, "result")
    os.makedirs(save_img_dir, exist_ok=True)

    # 手动绘制：只画边界框
    for res in results:
        img = res.orig_img.copy()
        boxes = res.boxes

        for box in boxes:
            # 坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 类别ID
            cls_id = int(box.cls[0])
            # 固定颜色
            color = get_fixed_color(cls_id)

            # 只画框：统一粗细（移除了文本和文本背景底色绘制）
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # 保存图片
        img_name = os.path.basename(res.path)
        save_path = os.path.join(save_img_dir, img_name)
        cv2.imwrite(save_path, img)

    print(f"\n✅ 推理完成！")
    print(f"🎉 可视化结果已保存到：\n{save_img_dir}")
    return results

# ------------------- 运行 -------------------
if __name__ == "__main__":
    import cv2  # 在这里导入防止提前报错
    yolo11_predict_and_visualize(
        weights=WEIGHTS_PATH,
        source=IMAGE_SOURCE,
        save_dir=OUTPUT_SAVE_DIR
    )