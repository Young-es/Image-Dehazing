import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

def get_atmospheric_light(img, patch_size=15):
    """
    步骤 1：估算大气光 A 就是和暗通道先验一样计算也是取得前0.1%的像素
    """
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    h, w = dark_channel.shape
    num_pixels = int(max(h * w * 0.001, 1))
    
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_pixels:]
    
    A = np.zeros(3)
    for idx in indices:
        r, c = divmod(idx, w)
        A = np.maximum(A, img[r, c, :])
    return A

def get_transmission_haze_line(img, A, num_clusters=500):
    """步骤 2 & 3：提取雾线并估算透射率"""
    h, w, c = img.shape
    # 1. 坐标系平移：将大气光 A 作为坐标系原点
    # 这样所有属于同一类物体的像素，在空间中就会形成一条从原点出发的射线
    I_A = img - A
    
    # 2.计算每个像素rgb的模长，r 代表像素受雾气影响的程度，r 越大说明颜色越纯，雾越少
    r = np.linalg.norm(I_A, axis=2)#线性代数工具箱的求模长
    r_flat = r.flatten() #做成一维数组
    
    # 防止除以零，设定一个极小的下限
    r_safe = np.maximum(r, 1e-6) #防止后续除以0
    
    # 计算方向向量 (剔除长度，只保留方向，即球面上的点)
    unit_vectors = I_A / r_safe[:, :, np.newaxis]
    unit_vectors_flat = unit_vectors.reshape(-1, 3)
    
    # 3. 寻找雾线：聚类 (K-Means)
    # 为了极速运算，我们只随机抽样 10000 个点来寻找这 500 根雾线 (聚类中心)
    sample_indices = np.random.choice(len(unit_vectors_flat), 10000, replace=False)#随机抽取
    samples = unit_vectors_flat[sample_indices]#抽取的1000个点的方向坐标拿到手
    
    # 使用 MiniBatchKMeans 进行高速聚类，使用机器学习算法快速聚类
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=3, random_state=42)
    kmeans.fit(samples)
    line_centers = kmeans.cluster_centers_
    
    # 4. 把全图所有的像素分配到离它角度最近的那根雾线上
    # 使用 KDTree 进行三维空间极速最近邻搜索
    tree = cKDTree(line_centers)  #用的KD树
    _, line_labels = tree.query(unit_vectors_flat)#用KD数进行快速分拣，拿到每个像素属于那个雾线
    
    # 5. 估算透射率 t
    t_flat = np.zeros_like(r_flat)
    
    for k in range(num_clusters):
        # 找出属于第 k 根雾线的所有像素的索引
        idx = np.where(line_labels == k)[0] #拿到每条雾线的各个像素
        if len(idx) > 0:
            # 取出这些像素的半径 r
            r_k = r_flat[idx]
            
            # 找到这条线上离原点最远的点 (即不受雾影响的最清晰像素)
            # 使用 99% 分位数代替 np.max，能够有效防止噪点把最大半径带偏
            r_max = np.percentile(r_k, 99) 
            
            # 透射率 t = 当前像素半径 / 最大半径
            t_k = r_k / np.maximum(r_max, 1e-6)
            t_flat[idx] = t_k
            
    # 将一维的透射率重新折叠回二维图像形状
    t_coarse = t_flat.reshape(h, w)
    
    # 透射率不能超过 1，也不能太小防止噪声放大
    return np.clip(t_coarse, 0.05, 1.0)

def guided_filter(I, p, r=41, eps=1e-4):
    """
    步骤 4：导向滤波 (平滑)
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
    t_clamped = np.maximum(t, t0)
    recovered = np.zeros_like(img)

    for i in range(3):
        recovered[:, :, i] = (img[:, :, i] - A[i]) / t_clamped + A[i]

    return np.clip(recovered, 0, 1)

def dehaze_haze_line(img_path):
    """
    雾线先验 (Haze-line Prior) 去雾主流程管线
    """
    # 1. 图像读取与归一化
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError("未找到图片，请检查路径是否正确！")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) / 255.0

    # 2. 核心算法流程

    A = get_atmospheric_light(img_rgb)
    

    # num_clusters=500 代表假设这幅画面由 500 种基础色彩的物体组成
    trans_coarse = get_transmission_haze_line(img_rgb, A, num_clusters=500)


    # 聚类产生的透射率图通常比较粗糙，导向滤波在这里必不可少
    trans_refined = guided_filter(img_gray, trans_coarse, r=41, eps=1e-4)


    result = recover_image(img_rgb, A, trans_refined, t0=0.1)

    # 3. 可视化展示
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(trans_coarse, cmap='gray')
    plt.title("2. Haze-Line Transmission")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(trans_refined, cmap='gray')
    plt.title("3. Refined Transmission")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(result)
    plt.title("4. Dehazed Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    dehaze_haze_line(r'D:\PythonProject1\projects\Image-Dehazing\testimg\Haze.jpg')