# A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior

基于颜色衰减先验的去雾算法相比于暗通道先验算法的不同之处，在于作者发现了另外一个图像中的统计规律：雾的浓度与亮度 $v$ 和饱和度 $s$ 的差成正比。

<div align="center">
  <img src="https://github.com/user-attachments/assets/85c58eff-1dbf-4532-ac2b-40491fd6b277" width="550" alt="Brightness and Saturation vs Haze Concentration" style="border-radius: 8px; margin: 15px 0;" />
</div>

我们来回顾一下雾图构成的公式，雾图主要有两个部分组成：一个部分是大气光成分，一个是物体在被反射和散射最后剩下进入手机中混合构成的图像就是雾图：

$$I(x) = J(x)t(x) + A(1 - t(x))$$

<div align="center">
  <img src="https://github.com/user-attachments/assets/bede5b3e-4f72-4f25-81eb-37f904499830" width="300" alt="Haze Imaging Equation" style="margin: 15px 0;" />
</div>

直接的衰减，会导致亮度变小，也就是：

$$J(x)t(x)$$


然而大气光的成分会增强亮度，而降低图像的饱和度。

所以作者提出了，我们可以根据饱和度和亮度的差值和雾的浓度的关系，在论文里作者提出了景深和雾的浓度和亮度与饱和度的关系：

$$d(x) = \theta_0 + \theta_1 v(x) + \theta_2 s(x) + \epsilon(x)$$

<div align="center">
  <img src="https://github.com/user-attachments/assets/c7a473c3-f57f-49d9-b436-499141fdb2e4" width="350" alt="Linear Depth Model" style="margin: 15px 0;" />
</div>

然后作者自己通过大量的训练，训练出了几个这个关系的固定参数：

$$\theta_0 = 0.121778,\quad \theta_1 = 0.959710,\quad \theta_2 = -0.780245,\quad \sigma = 0.041337$$

<div align="center">
  <img src="https://github.com/user-attachments/assets/95a454b6-28bd-4bb5-8132-f148363e8bb2" width="500" alt="Estimated Parameters" style="border-radius: 4px; margin: 15px 0;" />
</div>

然后后续的计算很关键的透射率，利用这样的一个衰减函数，根据算出来的图像每一处的景深图去求每一个部分的透射率。$\beta$ 是一个大气的散射系数，可调整：

$$t(x) = e^{-\beta d(x)}$$

<div align="center">
  <img src="https://github.com/user-attachments/assets/7e9c296c-879f-4646-840c-10a506a904a6" width="650" alt="Transmission Estimation" style="margin: 15px 0;" />
</div>

下面就是我跑出来的效果，可以看到效果还是很好的，他估计的透射率图比单纯的使用暗通道先验估计的透射率图更加精细、真实。

<div align="center">
  <img src="https://github.com/user-attachments/assets/ff3d2661-5f8f-46ad-b1a5-80b573c4f669" width="100%" alt="Dehazing Results Comparison" style="border-radius: 8px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
</div>
