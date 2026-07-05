# Non-Local Image Dehazing

然后我又去了解到了一种非局部去雾方法——基于雾线（Haze-Line）去雾的方法。

一段仿真下来，给我的感觉就是创造了一种新的计算透射率的方法，讲真的也就是换汤不换药。也许会使得透射率函数更加精细准确，但是有时候，过于精细也不是好事。并且论文里讲的是对天空更加的友好，但是计算参数的增多，也会对算法的性能产生影响，导致我们需要对具体情况具体分析，很不方便。

所以有时候不拘小节，允许偏差的存在也是一种选择，到达人们可以接受的偏差范围内！

<div align="center">
  <img src="https://github.com/user-attachments/assets/32823f7c-319d-456a-8fd0-c0de2bee9fb9" width="100%" alt="Haze Line Dehazing Result" style="border-radius: 8px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
</div>

### 雾线 (Haze-Line) 算法原理
**说白了就是一个把不同的雾的浓度下的透射率分了类别，然后每一个像素去找到我们分的类，作为自己这个地方的透射率**

**物理解释：** 
  把 RGB 色彩空间想象成一个三维的房间。大气光 $A$ 就是挂在房间里的“高亮灯泡”。在现实中，同一类物体（比如一片树林），因为距离镜头远近不同，受到的雾气干扰程度也不同。**雾气就像一股引力，把这些像素的颜色在 3D 空间里向着灯泡 $A$ 拉扯。**

结果是，这些像素在 RGB 空间里排成了一条笔直的射线，一端是真实的绿色，另一端直插大气光 $A$。这就是“雾线”。我们只要算出像素在这条线上的相对距离，就能直接得出透射率 $t$。

```python
def get_transmission_haze_line(img, A, num_clusters=500):
    """步骤 2 & 3：提取雾线并估算透射率"""
    h, w, c = img.shape
    
    # 1. 坐标系平移：将大气光 A 作为坐标系原点
    # 这样所有属于同一类物体的像素，在空间中就会形成一条从原点出发的射线
    I_A = img - A
    
    # 2. 计算每个像素 RGB 的模长
    # r 代表像素受雾气影响的程度，r 越大说明颜色越纯，雾越少
    r = np.linalg.norm(I_A, axis=2) # 线性代数工具箱的求模长
    r_flat = r.flatten() # 拍扁成一维数组
    
    # 防止除以零，设定一个极小的下限
    r_safe = np.maximum(r, 1e-6) # 防止后续除以 0
    
    # 计算方向向量 (剔除长度，只保留方向，即球面上的点)
    unit_vectors = I_A / r_safe[:, :, np.newaxis]
    unit_vectors_flat = unit_vectors.reshape(-1, 3)
    
    # 3. 寻找雾线：聚类 (K-Means)
    # 为了极速运算，我们只随机抽样 10000 个点来寻找这 500 根雾线 (聚类中心)
    sample_indices = np.random.choice(len(unit_vectors_flat), 10000, replace=False) # 随机抽取
    samples = unit_vectors_flat[sample_indices] # 抽取的10000个点的方向坐标拿到手
    
    # 使用 MiniBatchKMeans 进行高速聚类，使用机器学习算法快速聚类
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=3, random_state=42)
    kmeans.fit(samples)
    line_centers = kmeans.cluster_centers_
    
    # 4. 把全图所有的像素分配到离它角度最近的那根雾线上
    # 使用 KDTree 进行三维空间极速最近邻搜索
    tree = cKDTree(line_centers)  # 用的 KD 树
    _, line_labels = tree.query(unit_vectors_flat) # 用 KD 树进行快速分拣，拿到每个像素属于哪条雾线
    
    # 5. 估算透射率 t
    t_flat = np.zeros_like(r_flat)
    
    for k in range(num_clusters):
        # 找出属于第 k 根雾线的所有像素的索引
        idx = np.where(line_labels == k)[0] # 拿到每条雾线的各个像素
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

