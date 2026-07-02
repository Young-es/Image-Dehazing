import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_depth_map(img_rgb):
    """步骤 1：利用 CAP 线性模型计算景深图 (Depth Map)"""
    # 将图像转换为 HSV 颜色空间来提取 V (亮度) 和 S (饱和度)
    # 注意：OpenCV 的 HSV 中，V 的范围是 0-255，S 的范围也是 0-255，需要归一化到 [0, 1]
    img_hsv = cv2.cvtColor(np.float32(img_rgb), cv2.COLOR_RGB2HSV)

    # 提取 S 和 V 通道
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    # CAP 论文中通过机器学习训练出的参数
    theta_0 = 0.121779
    theta_1 = 0.959710
    theta_2 = -0.780245

    # 直接代入线性公式计算景深 d
    depth = theta_0 + theta_1 * v + theta_2 * s

    # 限制景深的范围，防止出现异常值
    return np.clip(depth, 0, 1)


def get_atmospheric_light_by_depth(img, depth_map):
    """步骤 2：利用景深图估算全局大气光 A
    和 DCP 的逻辑完全一样，只是我们这次找的是景深最深（最亮）的前 0.1% 的点"""
    h, w = depth_map.shape
    num_pixels = int(max(h * w * 0.001, 1))

    flat_depth = depth_map.flatten()
    # 景深越大，雾越浓，所以找景深最大的点
    indices = np.argsort(flat_depth)[-num_pixels:]

    A = np.zeros(3)
    for idx in indices:
        r, c = divmod(idx, w)
        A = np.maximum(A, img[r, c, :])
    return A


def get_transmission_cap(depth_map, beta=1.0):
    """步骤 3：计算透射率 (Transmission t)利用指数衰减公式：t = exp(-beta * d)"""
    # 这里的 beta 是散射系数，可以调节去雾的强度
    # beta 越大，算出来的透射率 t 越小，去雾越强烈
    transmission = np.exp(-beta * depth_map)
    return transmission


def guided_filter(I, p, r=41, eps=1e-4):
    """步骤 4：引导滤波"""
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
    """步骤 5：利用物理公式恢复无雾图像 """
    t_clamped = np.maximum(t, t0)
    recovered = np.zeros_like(img)
    for i in range(3):
        recovered[:, :, i] = (img[:, :, i] - A[i]) / t_clamped + A[i]
    return np.clip(recovered, 0, 1)


def dehaze_cap(img_path):
    """CAP 去雾主流程管线"""
    # 1. 读取图像并归一化
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError("未找到图片，请检查路径是否正确！")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0

    # 2. 核心算法流程
    depth_map = get_depth_map(img_rgb)
    A = get_atmospheric_light_by_depth(img_rgb, depth_map)

    # beta=1.0 是默认值，如果是特别浓的雾，可以尝试调大 (比如 1.2, 1.5)
    trans_coarse = get_transmission_cap(depth_map, beta=1.0)

    # 虽然 CAP 是逐像素计算的，没有严重的方块效应，
    # 但景深图本身可能有些噪点，使用导向滤波平滑一下边缘效果会更好
    trans_refined = guided_filter(img_gray, trans_coarse, r=41, eps=1e-4)

    # 恢复图像 (同样保留 t0=0.1 防止噪声爆炸)
    result = recover_image(img_rgb, A, trans_refined, t0=0.1)

    # 3. 可视化展示
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(depth_map, cmap='gray')
    plt.title("2. Estimated Depth Map")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(trans_refined, cmap='gray')
    plt.title("3. Refined Transmission")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(result)
    plt.title("4. CAP Dehazed Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dehaze_cap(r'D:\PythonProject1\projects\Image-Dehazing\testimg\tiananmen1.png')