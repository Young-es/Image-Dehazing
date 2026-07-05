# Image Dehazing:Study Notes & Implementation Reflections
1. 直方图均衡化去雾
2. 暗通道先验算法 (Dark Channel Prior, DCP)
3. 颜色衰减先验算法 (Color Attenuation Prior, CAP)
4. 雾线去雾算法 (Haze-line Prior)
5. Model-driven 深度学习去雾 (以 AOD-Net 为代表)

**1.直方图均衡化没什么好说的，就是按照每个灰度值出现的次数去计算概率，然后这个概率按照0~255个灰度级去重新分配每个像素的灰度级，提升对比度**
**2.暗通道先验很经典，详细的学习笔记在我的DCP文件中**
**3.颜色衰减先验是在暗通道先验的基础上去更新了计算透射率的方式，也是根据统计规律，详细的学习心得也在我的Color_Prior文件夹中**
**4.雾线先验也是提出一种新的计算透射率的方式，把每个地方的透射率分门别类学习心得在我的HIP文件夹中**
**5.AOD-Net比较经典，跑出来的效果也比较好，我主要是复现别人的工作fork在，尝试训练了一下**
