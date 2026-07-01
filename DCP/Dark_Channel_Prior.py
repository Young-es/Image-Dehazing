import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dark_channel(img, patch_size=15):
    """
    步骤 1：计算暗通道图 (Dark Channel)
    """
    # 提取 RGB 三个通道中的最小值
    min_channel = np.min(img, axis=2)
    # 使用形态学腐蚀（等价于局部窗口内的最小值滤波）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def get_atmospheric_light(img, dark_channel):
    """步骤 2：估算全局大气光 (Atmospheric Light A)"""
    h, w = dark_channel.shape
    num_pixels = int(max(h * w * 0.001, 1))  # 取暗通道中最亮的前 0.1% 的像素

    # 展平数组并获取最亮像素的索引
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_pixels:]

    A = np.zeros(3)
    for idx in indices:
        # 将 1 维索引转回 2 维坐标
        r, c = divmod(idx, w)
        # 在原图中对应的位置寻找最高亮度值作为大气光
        A = np.maximum(A, img[r, c, :])

    return A


def get_transmission(img, A, patch_size=15, w=0.95):
    """
    步骤 3：粗略估算透射率 (Transmission t)
    """
    # 将图像归一化到大气光 A
    normalized_img = img / A
    dark = get_dark_channel(normalized_img, patch_size)

    # w=0.95 是保留极少量雾气的参数，让图片看起来更自然，远景有一定的景深感
    transmission = np.maximum(1 - w * dark, 0)
    return transmission


def guided_filter(I, p, r=41, eps=1e-4):
    """
    步骤 4：导向滤波 (Guided Filter) - 用于细化透射率图边缘，消除块状马赛克
    """
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    mean_II = cv2.boxFilter(I * I, -1, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    q = mean_a * I + mean_b
    return q


def recover_image(img, A, t, t0=0.1):
    """
    步骤 5：利用物理公式恢复无雾图像
    """
    # 设定透射率下限 t0，防止除以 0 导致数值溢出
    t_clamped = np.maximum(t, t0)
    recovered = np.zeros_like(img)

    for i in range(3):
        # 物理退化反解公式：J = (I - A) / t + A
        recovered[:, :, i] = (img[:, :, i] - A[i]) / t_clamped + A[i]

    # 将像素值截断到 [0, 1] 之间，防止产生怪异的噪点伪影
    return np.clip(recovered, 0, 1)


def dehaze(img_path):
    """
    去雾主流程管线
    """
    # 1. 读取图像并转为 RGB 浮点型，归一化到 [0, 1]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError("未找到图片，请检查路径是否正确！")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0

    # 提取一张灰度图，作为导向滤波的“引导图”
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0

    # 2. 核心算法流程
    patch_size = 15
    dark_channel = get_dark_channel(img_rgb, patch_size)
    A = get_atmospheric_light(img_rgb, dark_channel)
    trans_coarse = get_transmission(img_rgb, A, patch_size, w=0.95)

    # 使用导向滤波平滑透射率图的边缘
    trans_refined = guided_filter(img_gray, trans_coarse, r=41, eps=1e-4)

    # 恢复图像
    result = recover_image(img_rgb, A, trans_refined, t0=0.1)

    # 3. 可视化展示
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(trans_coarse, cmap='gray')
    plt.title("2. Coarse Transmission Map")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(trans_refined, cmap='gray')
    plt.title("3. Refined Transmission Map")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(result)
    plt.title("4. Dehazed Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 绝对路径运行
    dehaze(r'/Image-Dehazing/testimg\canon3.bmp')