# 直方图均衡化去雾尝试
# 包括全局直方图均衡化和自适应局部直方图均衡化
import cv2
import matplotlib.pyplot as plt


def dehaze_by_histogram(img_path):
    # 1. 读取有雾图像并转为 RGB 供展示
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 方法一：全局直方图均衡化 (Global HE)
    # 转到 YCrCb 色彩空间，只对亮度通道（Y）做均衡化，避免色彩失真
    # ---------------------------------------------------------
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(img_ycrcb))
    channels[0] = cv2.equalizeHist(channels[0])
    img_ghe_rgb = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2RGB)

    # 方法二：自适应直方图均衡化 (CLAHE)
    # 分块限制对比度拉伸，工业界更常用的边缘细节增强管线
    img_ycrcb_clahe = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    channels_clahe = list(cv2.split(img_ycrcb_clahe))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels_clahe[0] = clahe.apply(channels_clahe[0])
    img_clahe_rgb = cv2.cvtColor(cv2.merge(channels_clahe), cv2.COLOR_YCrCb2RGB)

    # 3. 结果展示
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_ghe_rgb)
    plt.title("2. Global HE")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_clahe_rgb)
    plt.title("3. CLAHE")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dehaze_by_histogram(r'D:\PythonProject1\projects\Image-Dehazing\testimg\canon3.bmp')
