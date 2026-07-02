### 1.

<div align="center">
  <img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/1f7a15bd-25b1-4423-8a48-7d48a40caa3c" />
  <br>
  <img width="399" height="66" alt="image" src="https://github.com/user-attachments/assets/0c084751-a480-4825-ba5e-348d06f36b27" />
</div>

这一段代码主要实现的是这个公式，暗通道先验，我的理解是，每一张清晰图片中(no sky)的大部分局部区域，至少有一个颜色通道的像素值很低，甚至接近于0，基于这样的一个先验条件，去求解大气散射模型的透射率，会方便很多，我在这个函数里用的是腐蚀函数去计算局部窗口下的最小值替换操作。

---

### 2.

<div align="center">
  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/3b106619-7b11-4fc1-80c3-447c24e44b10" />
  <br>
  <img width="598" height="382" alt="image" src="https://github.com/user-attachments/assets/c848ae5f-8e60-49ee-b831-5b8ab6c9863e" />
</div>

这里是一个估计大气光的函数，何凯明老师认为，估计大气光，我们可以观测刚刚计算的暗通道中的前0.1%的点，然后这些点对应到原图I中，再去找最亮的点做为A的估计值，对于去计算大气的估计值，我们为什么要去借助暗通道，我个人认为，非雾密集区域，像上图中的白色小车，其周围像素很可能就满足暗通道先验，这使得小车只有最中心的像素在暗通道是比较亮的，很可能就比雾密集区域的亮度低了。但对于一大片白色的物体，那就没法了。

---

### 3.

<div align="center">
  <img width="795" height="372" alt="image" src="https://github.com/user-attachments/assets/7ba45aea-26b1-4423-a9f9-b5a44fdafc89" />
  <br>
  <img width="486" height="80" alt="image" src="https://github.com/user-attachments/assets/adbc798b-3bba-41df-a07b-71f8bb09e2f9" />
</div>

这里是根据暗通道先验去计算的大气散射模型的透射率函数，同样的也是利用了我写的第一个的计算get_dark_channel函数去求I/A的暗通道就可以了，A里面储存了我们刚刚求出来的原图中最亮最接近与大气的颜色，何凯明老师在这里提到的是全局均匀大气光

---

### 4.

<div align="center">
  <img width="650" height="563" alt="image" src="https://github.com/user-attachments/assets/e553c0e7-ad00-4f1a-bf32-bed641f3c724" />
  <br>
  <img width="757" height="489" alt="image" src="https://github.com/user-attachments/assets/a54d4d76-a732-4834-8707-9a1040c474f6" />
</div>

第四个部分也是何凯明老师提出来的引导滤波，是为了细化透射率图片，前面直接用15×15的局部去得到的透射率图会有方形的马赛克，所以在这里我们用图像的灰度图去做为引导图，去精细化我们的传输函数图，引导滤波的核心假设是：最终精细的透射率 $q$ 和引导图 $I$ 满足线性关系（$q = aI + b$）。根据何凯明论文中得到的a和b的公式可以求解。

---

### 5.

<div align="center">
  <img width="840" height="394" alt="image" src="https://github.com/user-attachments/assets/6abfa30d-d391-4728-a37f-566165d3d16e" />
  <br>
  <img width="319" height="90" alt="image" src="https://github.com/user-attachments/assets/8a57e00d-6f58-4fea-8f54-fa517cbe96a7" />
</div>

这个公式中最巧妙的地方在于t_clamped = np.maximum(t, t0)，t 是刚才导向滤波修好的精细透射率矩阵。如果某些地方雾特别浓，t 的值会逼近 0。
根据公式，透射率是作为分母的。如果分母是 0.001，那么原本极其微弱的高频噪声或背景杂讯，就会被放大 1000 倍。
这一行通过 np.maximum(t, 0.1) 强制规定：全矩阵哪怕透射率再低，也只能算 0.1。这样就把噪声放大系数牢牢限制在 10 倍以内，保住了画面不会出现可怕的彩色杂斑。

---

### 跑出来的实验结果

<div align="center">
  <img width="1630" height="488" alt="image" src="https://github.com/user-attachments/assets/92751cd7-1ede-4249-bb7b-e4d2a979c917" />
</div>
