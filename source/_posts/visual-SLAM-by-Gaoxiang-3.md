---
title: visual SLAM by Gaoxiang(3)
mathjax: true
tags:
  - visual SLAM
  - Linux
  - C++11
  - Computer vision
  - Sophus
categories: 科研记录
abbrlink: 211277813
date: 2019-03-26 17:14:25
---

​                          

本次课程主要研究李群和李代数(Lie Group, Lie Algebra)，主要的目的是为了能够相机得旋转和平移进行微调。因为相机的运动估计可能不准确，而无法对旋转矩阵加上微小量之后依然是旋转矩阵（旋转矩阵无法定义加法，如果用四元数，必须是单位四元数，那么也无法定义加法）。李群李代数与后面的优化，流形都会有很大的联系。

>在视觉SLAM中，相机的位姿是未知的，而我们需要解决什么样的相机位姿最符合当前观测数据这样的问题。一种典型的方式是把其构建成一个优化问题，求解最优的$R$和$t$，使得误差最小化。
>
>由于旋转矩阵自身带有约束，即必须正交且行列式为1，因此作为优化变量会引入额外的约束，使得优化变得困难。而通过李群李代数的转换关系，可以顺利求导，把位姿估计变成无约束的优化问题。

 

## 群

群(Group)是一种集合加上一种运算的代数结构，满足封闭性，结合律，**幺元**，逆。其中幺元可以认为是单位元，就是与其他元素作用不改变这个元素，逆是元素和和它的逆进行运算后得到了幺元。

三维旋转矩阵构成了三维正交群(special orthogonal group)
$$
SO(3) = \left \{ R \in {\mathbb R}^{3 \times 3} | RR^{T} =I, \det (R)=1 \right \}
$$

三维变换矩阵构成了特殊欧式群(special euclidean group)
$$
SE(3) = 
\left \{
T= 
\begin{bmatrix}
R & t\\
O^{T} & 1
\end{bmatrix}
\in {\mathbb R}^{4 \times 4} |
R \in SO(3), t \in {\mathbb R}^{3} 
\right \}
$$

旋转矩阵集合与矩阵乘法构成群，变换矩阵集合与矩阵乘法也构成了群，因此称它们为旋转矩阵群和变换矩阵群。

群结构保证了在群上的运算具有良好的性质。

<!--more-->



## 李群与李代数

### 李群

具有连续（光滑）性质的群；

既是群也是流形；

直观上看，一个刚体能够连续地在空间中运动，因此$SO(3)$和$SE(3)$都是李群，然而，它们都没有定义加法，所以很难进行取极限和求导等操作；

### 李代数

与李代数对应的一种结构，位于向量空间（李群单位元处的正切空间）$\mathfrak so(3)$，$\mathfrak se(3)$

从旋转矩阵引出旋转正交群的李代数：

对于相机的连续运动，旋转矩阵也随时间变化，则有：
$$
\begin{align*}
& R(t)R(t)^{T} = I \\
& \dot{R}(t)R(t)^{T}+R(t)\dot{R}(t)^{T} = 0  \quad \quad 对时间求导\\
& \dot{R}(t)R(t)^{T} = - (\dot{R}(t)R(t)^{T})^{T} \quad \quad 反对称矩阵\\
& 记\dot{R}(t)R(t)^{T} = \phi (t)^{\wedge} \implies \dot{R}(t) = \phi (t)^{\wedge} R(t)
\end{align*}
$$
符号$\wedge$看作是反对称矩阵的符号，在这里是指将向量$\phi$变成了反对称矩阵，这是由叉乘引申而来，在第二讲有提过。反过来符号$\vee$代表反对称矩阵到向量的变换。

上面的式子表示，对旋转矩阵求导，就是在其左侧乘以一个$\phi (t)$，类似于指数函数的求导。

下面进行进一步地近似，假设在单位元附近，$t_{0}=0, R(0)=I$，则：
$$
R(t) \approx R(t_{0}) + \dot {R}(t_{0})(t-t_{0}) = I + \phi (t_{0})^{\wedge} (t) \quad \quad 将R在t_{0}进行泰勒展开，并忽略二次及以上高阶项 
$$
进一步假设，在$t_{0}$附近，$\phi$不变，则$\dot {R} (t) = \phi (t_{0}) ^{\wedge} R(t) = \phi _{0} ^{\wedge} R(t)$，再根据初值条件，得出：
$$
R(t) = \exp( \phi _{0} ^{\wedge} t)
$$
在泰勒展开那一步可以看出，$\phi$反映的是一阶导数的性质，位于旋转正交群的正切空间上（tangent space，切平面上）。

上述证明提供了一种思路，可能不太严谨。实际上可以证明最后得出的式子在任意时间都适用，且该关系称为指数映射（exponential map），$\phi$称为$SO(3)$对应的李代数$\mathfrak so(3)$。

**李群是高维空间的低维曲面，或者说低维流形，在流形原点附近的切空间上的任意一个点，是李代数，可以通过指数映射映回李群上。李代数描述了李群单位元附近的正切空间的性质。**
$$
\begin{align*}
& 李代数由一个集合\mathbb V，一个数域\mathbb F，和一个二元运算[,]（李括号，直观上说表示了两个元素的差异）组成。如果满足下面的四条性质，称\\
& (\mathbb V, \mathbb F, [,])为一个李代数，记作 \mathfrak g。\\
&1. 封闭性 \quad \quad \forall {\bf X, Y}\in \mathbb V, [{\bf X,Y}] \in \mathbb V\\
&2. 双线性 \quad \quad \forall {\bf X, Y}\in \mathbb V, a,b \in \mathbb F,有：
[a {\bf X} + b {\bf Y}, {\bf Z}] =a[{\bf X,Z}]+b[{\bf Y, Z}],[{\bf Z},a{\bf X}+b{\bf Y}] =a[{\bf Z,X}]+b[{\bf Z,Y}] \\
&3. 自反性 \quad \quad \forall {\bf X} \in \mathbb V, [{\bf X,X}]={\bf 0}\\
&4. 雅可比等价 \quad \quad \forall {\bf X,Y,Z} \in \mathbb V,[{\bf X},[{\bf Y,Z}]]+[{\bf Z},[{\bf Y,X}]]+[{\bf Y},[{\bf Z,X}]]= {\bf 0}


\end{align*}
$$



李代数$\mathfrak so(3)$可以看成是三维空间向量和叉积运算构成的，$so(3)=\left \{ \phi \in \mathbb R^{3}, \Phi = \phi ^{\wedge} \in \mathbb R^{3 \times 3} \right \}$，其中：
$$
\Phi = \phi ^{\wedge} =
\begin{bmatrix}
0 & - \phi_{3} & \phi _{2}\\
\phi _{3} & 0 & -\phi _{1}\\
-\phi _{2} & \phi _{1} & 0
\end{bmatrix}
\in \mathbb R^{3 \times 3}
$$
李括号$[\phi_{1},\phi_{2}] = (\Phi_{1} \Phi_{2} - \Phi_{2} \Phi_{1})^{\vee}$,容易验证此李括号满足上述四条性质。



对于变换矩阵的特殊欧式群$SE(3)$，也有对应的李代数$\mathfrak se(3)$(6维的向量)
$$
\mathfrak se(3) = \left \{
\xi = 
\begin{bmatrix}
\rho \\
\phi
\end{bmatrix}
\in \mathbb R^{6}, \rho \in \mathbb R^{3}, \phi \in \mathfrak se(3),
\xi ^{\wedge} = 
\begin{bmatrix}
\phi ^{\wedge} & \rho \\
{\bf 0}^{T} & 0
\end{bmatrix}
\in \mathbb R^{4 \times 4}
\right \}
$$
此时还是以用符号$\wedge$来表示向量到矩阵的变换，只不过不再是限制于反对称矩阵。
$$
\begin{align*}
& 设变换矩阵g(t)=
\begin{bmatrix}
R & \alpha \\
{\bf 0}^{T} & 1
\end{bmatrix} 
，则g(t)^{-1} = 
\begin{bmatrix}
R^{T} & -R^{T}\alpha \\
{\bf 0}^{T} & 1
\end{bmatrix}\\
&有\dot{g}(t)g(t)^{-1}=
\begin{bmatrix}
\dot{R}R^{T} & \alpha- \dot{R}R^{T}\alpha \\
{\bf 0}^{T} & 0
\end{bmatrix}
= 
\begin{bmatrix}
\omega^{\wedge} & v\\
{\bf 0}^{T} &0
\end{bmatrix}
，其中，\omega^{\wedge} \in \mathbb R^{3 \times 3},v \in \mathbb R^{3},记
\xi^{\wedge}=\dot{g}(t)g(t)^{-1}\\
& (\xi^{\wedge})^{\vee}=
\begin{bmatrix}
v \\
\omega
\end{bmatrix} \in \mathbb R^{6}
\\
&\dot{g}(t)=(\dot{g}(t)g(t)^{-1})g(t)=\xi^{\wedge}g(t),则g(t)=\exp(\xi^{\wedge}),假设g(0)=I
\end{align*}
$$


李括号$[\xi _{1}, \xi _{2}] = (\xi _{1} ^{\wedge} \xi_{2} ^{\wedge} - \xi _{2} ^{\wedge} \xi_{1} ^{\wedge}) ^{\vee}$



## 指数映射和对数映射

指数映射反映了李代数到李群的关系，对于旋转矩阵$R$，有$R=\exp (\phi ^{\wedge})=\sum_{n=0}^{\infty} \frac{1}{n!}(\phi ^{\wedge})^{n}$

为了研究的方便，先将$\phi$写成旋转向量的形式，即设$\phi = \theta \vec{a}$,其中$\vec{a}$是单位向量，而且具有以下性质
$$
\begin{align*}
&1.\vec{a}^{\wedge} \vec{a}^{\wedge} = \vec {a} \vec{a}^{T} - I \\
&2.\vec {a}^{\wedge} \vec {a}^{\wedge} \vec {a}^{\wedge} = -\vec {a}^{\wedge}
\end{align*}
$$
下面利用上述性质对$\exp (\phi)^{\wedge}$进行Taylor展开
$$
\begin{align*}
\exp (\phi)^{\wedge} & =\exp (\theta \vec{a}) = \sum _{n=0}^{\infty}\frac{1}{n!}(\theta \vec{a}^{\wedge})^{n}\\
                     & = I+\theta \vec{a}^{\wedge}+\frac{1}{2!}\theta ^{2} \vec{a}^{\wedge}\vec{a}^{\wedge}+\frac{1}{3!} \theta ^{3} \vec{a}^{\wedge}\vec{a}^{\wedge}\vec{a}^{\wedge}+\frac{1}{4!} \theta ^{4} \vec{a}^{\wedge}\vec{a}^{\wedge}\vec{a}^{\wedge}\vec{a}^{\wedge}+ \cdots\\
                     & = \vec{a}\vec{a}^{T}-\vec{a}^{\wedge}\vec{a}^{\wedge}+\theta \vec{a}^{\wedge}+\frac{1}{2!}\theta ^{2}\vec{a}^{\wedge}\vec{a}^{\wedge}-\frac{1}{3!} \theta^{3}\vec{a}^{\wedge}-\frac{1}{4!}\theta ^{4}\vec{a}^{\wedge}\vec{a}^{\wedge}+\cdots \\
                     & = \vec{a}\vec{a}^{T}+(\theta -\frac{1}{3!}\theta^{3}+\frac{1}{5!}\theta^{5}-\cdots)\vec{a}^{\wedge}-(1-\frac{1}{2!}\theta^{2}+\frac{1}{4!}\theta^{4}-\cdots)\vec{a}^{\wedge}\vec{a}^{\wedge}\\
                     & = \vec{a}^{\wedge}\vec{a}^{\wedge}+I+\sin \theta \vec{a}^{\wedge}-\cos \theta \vec{a}^{\wedge}\vec{a}^{\wedge}\\
                     & = (1-\cos \theta)\vec{a}^{\wedge}\vec{a}^{\wedge}+I+\sin \theta \vec{a}^{\wedge}\\
                     & = \cos \theta I+(1-\cos \theta)\vec{a}\vec{a}^{T}+\sin \theta \vec{a}^{\wedge}
                 
\end{align*}
$$
这进一步说明了李代数$\mathfrak so(3)$的物理意义确实就是旋转向量。

反之，给定旋转矩阵亦可以求出对应的李代数，即对数映射$\phi = \ln(R)^{\vee}$。不过实际情况下可以通过旋转矩阵到向量的公式来得出李代数。

同理，可以得到$\mathfrak se(3)$到$SE(3)$的指数映射（具体推导过程会在作业中展示）
$$
\begin{align*}
\exp(\xi^{\wedge}) & = 
\begin{bmatrix}
\sum _{n=0}^{\infty}\frac{1}{n!}(\phi^{\wedge})^{n} & \sum _{n=0}^{\infty} \frac{1}{(n+1)!}(\phi ^{\wedge})^{n} \rho \\
{\bf 0}^{T} & 1
\end{bmatrix}
\\
& \triangleq 
\begin{bmatrix}
R & J\rho \\
{\bf 0}^{T} & 1
\end{bmatrix}
\\
& = T
\end{align*}
$$
其中$J$为$SE(3)$的雅可比矩阵，$J=\frac{\sin \theta}{\theta} I + (1-\frac{\sin \theta} {\theta}) \vec{a} \vec{a}^{T} + \frac{1-\cos \theta} {\theta} \vec{a}^{\wedge}$

注意，这里的平移部分与变换矩阵的平移部分不完全相同，可以看出，指数映射会对李代数中的平移部分进行线性变换后才得到了真正的平移部分。

![李群李代数的对应关系](visual-SLAM-by-Gaoxiang-3\lie_group_and_lie_algebra.png)



## 李代数求导与扰动模型

前面说过，视觉SLAM应用李群李代数的最初目的就是为了对相机的位姿进行优化。因为相机在观测世界的时候，会不可避免地引入噪声，而我们优化的目的就是会取N个观测，然后对其进行误差最小化，得到一个在这么多N个观测的过程中最优的变换关系。因此，针对优化（一般是最小化问题）问题，我们常用的手段就是求导，此时李代数的功能就体现出来了，因为李代数具有良好的加法运算，可以进行无约束优化问题分析。

不过问题是，李代数上的加法并不会对应李群上的乘法，即$\exp(\phi_{1}^{\wedge}) \exp(\phi_{2}^{\wedge}) \neq \exp ((\phi_{1}+\phi_{2})^{\wedge})$

该关系由BCH公式（Baker-Campbell-Hausdorff）给出：
$$
\ln (\exp (A) \exp(B)) =
A +B + \frac{1}{2} [A,B] + \frac{1}{12}[A,[A,B]] - \frac{1}{12}[B.[A,B]]+\cdots
$$
如果其中一个量为小量，则有下列的近似表达关系：
$$
\ln (\exp (\phi_{1}^{\wedge}) \exp (\phi_{2}^{\wedge}))^{\vee} \approx
\begin{cases}
J_{l} (\phi _{2})^{-1} \phi_{1}+\phi_{2} \quad \quad \text{if  $\phi_{1}$is small}\\
J_{r}(\phi_{1})^{-1} \phi_{2} + \phi_{1} \quad \quad \text{if  $\phi_{2}$is small}
\end{cases}
$$
其中：
$$
\begin{align*}
& J_{l}=J=\frac{\sin \theta}{\theta} I + (1-\frac{\sin \theta} {\theta}) \vec{a} \vec{a}^{T} + \frac{1-\cos \theta} {\theta} \vec{a}^{\wedge}  \quad左雅可比\\
& J_{l}^{-1} = \frac{\theta}{2}\cot \frac{\theta}{2}I+(1-\frac{\theta}{2}\cot \frac{\theta}{2})\vec{a}\vec{a}^{T}-\frac{\theta}{2}\vec{a}^{\wedge}\\
& J_{r}(\phi) = J_{l}(-\phi) \quad 右雅可比

\end{align*}
$$

一般来说，利用$T_{cw}$时会进行左乘，利用$T_{wc}$时会进行右乘，以左乘为例，直观的写法是
$$
\begin{align*}
& \exp (\Delta \phi ^{\wedge}) \exp(\phi ^{\wedge}) = \exp ((J_{l}(\phi)^{-1}\Delta \phi +\phi)^{\wedge}) \quad \quad 李群上的微小量乘法，李代数上的加法相差雅可比矩阵的逆\\
& \exp((\phi + \Delta \phi)^{\wedge})=\exp ((J_{l}\Delta \phi)^{\wedge})\exp(\phi^{\wedge})=\exp(\phi^{\wedge}) \exp((J_{r}\Delta \phi)^{\wedge})
\quad \quad 李代数上的微小量加法，李群上要乘上雅可比矩阵
\end{align*}
$$


对于$SE(3)$和$\mathfrak se(3)$，关系要复杂一些（雅可比矩阵较为复杂，是个$6 \times 6$矩阵，但在实际计算中不用到该雅可比）：
$$
\begin{align*}
& \exp(\Delta \xi ^{\wedge}) \exp(\xi ^{\wedge}) \approx \exp ((\mathcal J_{l}^{-1}\Delta \xi +\xi)^{\wedge})\\
& \exp(\xi ^{\wedge}) \exp (\Delta \xi^{\wedge}) \approx \exp ((\mathcal J_{r}^{-1} \Delta \xi+\xi)^{\wedge})

\end{align*}
$$


有了上述公示后，开始对旋转矩阵进行求导，这里有两种方法，一种是导数定义求导，一种是通过扰动模型进行求导，实际中，扰动模型因为形式更加简单，因此采用的更多，先不严谨地记旋转后的点关于旋转矩阵的导数的求导式为 $\frac{\partial (Rp)}{\partial R}$

导数模型：
$$
\begin{align*}
\frac{\partial(\exp (\phi^{\wedge})p)}{\partial \phi} & = \lim_{\delta \phi \to 0} \frac{\exp((\phi+\delta \phi)^{\wedge})p-\exp(\phi^{\wedge})p}{\delta \phi} \\
                                                      & = \lim_{\delta \phi \to 0} \frac{\exp ((J_{l}\delta \phi)^{\wedge})\exp(\phi^{\wedge})p-\exp (\phi^{\wedge})p}{\delta \phi}\\
                                                      & \approx \lim_{\delta \phi \to 0} \frac{(I+(J_{l}\delta \phi)^{\wedge})\exp(\phi^{\wedge})p-\exp(\phi^{\wedge})p}{\delta \phi} \\
                                                      & = \lim_{\delta \phi \to 0} \frac{(J_{l}\delta \phi)^{\wedge}\exp(\phi^{\wedge})p}{\delta \phi}\\
                                                      & = \lim_{\delta \phi \to 0} \frac{-(\exp(\phi^{\wedge})p)^{\wedge} J_{l}\delta \phi}{\delta \phi} \\
                                                      & = -(Rp)^{\wedge}J_{l}

\end{align*}
$$
扰动模型（左乘微小量）：
$$
\begin{align*}
\frac{\partial(Rp)}{\partial R} & = \lim_{\delta \phi \to 0} \frac{\exp((\delta \phi)^{\wedge})\exp(\phi^{\wedge})p-\exp(\phi^{\wedge})p}{\delta \phi}\\
                                & \approx \lim_{\delta \phi \to 0} \frac{(I+(\delta \phi)^{\wedge})\exp(\phi^{\wedge})p-\exp(\phi^{\wedge})p}{\delta \phi}\\
                                & = \lim_{\delta \phi \to 0} \frac{(\delta \phi)^{\wedge}Rp}{\delta \phi} =  -(Rp)^{\wedge}
\end{align*} 
$$
同理可得，$SE(3)$上的扰动模型（左乘微小量）为：
$$
\frac{\partial (Tp)}{\partial \delta \xi}=
\begin{bmatrix}
I & -(Rp+t)^{\wedge}\\
{\bf 0}^{T} & {\bf 0}^{T}
\end{bmatrix}
\triangleq (Tp)^{\odot}
$$



## 相似变换

对于单目视觉，由于存在尺度不确定性，因此不能使用$SE(3)$来表达位姿变化，而是利用相似变换群$Sim(3)$，也就是说要加一个尺度因子$s$，这个尺度因子会同时作用在变换的点$p$上，对其进行缩放，也就是在相机坐标系下进行了一次相似变换，而不是欧式变换，即：$p^{'} = sRp+t$
$$
Sim(3)=\left\{\left[ \begin{array}{lll}{\boldsymbol{S}=} & {\boldsymbol{s} \boldsymbol{R}} & {\boldsymbol{t}} \\ {\boldsymbol{0}^{T}} & {1}\end{array}\right] \in \mathbb{R}^{4 \times 4}\right\}
$$

$$
\mathfrak sim(3)=\left\{\zeta | \zeta=\left[ \begin{array}{l}{\rho} \\ {\phi} \\ {\sigma}\end{array}\right] \in \mathbb{R}^{7}, \zeta^{\wedge}=\left[ \begin{array}{cc}{\sigma I+\phi^{\wedge}} & {\rho} \\ {0^{T}} & {0}\end{array}\right] \in \mathbb{R}^{4 \times 4}\right\}
$$

$$
\exp \left(\zeta^{\wedge}\right)=\left[ \begin{array}{cc}{e^{\sigma} \exp \left(\phi^{\wedge}\right)} & {J_{s} \rho} \\ {0^{T}} & {1}\end{array}\right]
$$

$$
\begin{aligned} J_{s}=& \frac{e^{\sigma}-1}{\sigma} I+\frac{\sigma e^{\sigma} \sin \theta+\left(1-e^{\sigma} \cos \theta\right) \theta}{\sigma^{2}+\theta^{2}} a^{\wedge} \\ &+\left(\frac{e^{\sigma}-1}{\sigma}-\frac{\left(e^{\sigma} \cos \theta-1\right) \sigma+\left(e^{\sigma} \sin \theta\right) \theta}{\sigma^{2}+\theta^{2}}\right) a^{\wedge} a^{\wedge} \end{aligned}
$$

$$
s=e^{\sigma}, \boldsymbol{R}=\exp \left(\boldsymbol{\phi}^{\wedge}\right), \boldsymbol{t}=\boldsymbol{J}_{s} \boldsymbol{\rho}
$$



对于$Sim(3)$的求导，利用左扰动模型和BCH近似（这里的BCH近似与$SE(3)$公式不同）。假设点$p$经过相似变换$Sp$后，相对于$S$的导数为：
$$
\frac{\partial S p}{\partial \zeta}=\left[ \begin{array}{ccc}{\boldsymbol{I}} & {-\boldsymbol{q}^{\wedge}} & {\boldsymbol{q}} \\ {\mathbf{0}^{T}} & {\mathbf{0}^{T}} & {0}\end{array}\right]
$$

其中$q$是$Sp$的前三维向量，最后的形式应该是$4 \times 7$的雅可比矩阵。

有关相似变换群的更为详细的理解和运用，等后面进行实际应用时再说，毕竟库已经提供好了，而且推导过程也与$SE(3)$类似。

 


## 作业与实践



### 群的性质

群要满足封闭性，结合律，幺元和逆这四个性质。其中，满足前两个性质的叫半群，满足前三个性质的叫有单位元的半群，若满足了上述四个性质，还具有交换律的叫做阿贝尔群。

对于$\left \{ \mathbb Z, + \right \}$封闭性和结合律显然满足，幺元是0，逆为自身的相反数，因此是群，而且是阿贝尔群。

对于$\left \{ \mathbb N, +\right \}$，前三个性质都满足，幺元是0，但是除了0之外，其他的元素不存在逆，因此不是群，是有单位元的半群。



### 验证向量叉乘的李代数性质

设${\bf X}=a_{1}\vec{i}+b_{1}\vec{j}+c_{1}\vec{k}, {\bf Y}=a_{2}\vec{i}+b_{2}\vec{j}+c_{2}\vec{k}, {\bf Z}=a_{3}\vec{i}+b_{3}\vec{j}+z_{3}\vec{k}$

封闭性：
$$
[{\bf X},{\bf Y}] = {\bf X} \times {\bf Y} \in \mathbb R^{3}
$$
双线性：
$$
\begin{align*}
& [a{\bf X}+b{\bf Y},{\bf Z}]=(a{\bf X}+b{\bf Y}) \times {\bf Z}=a{\bf X}\times {\bf Z}+b{\bf Y} \times {\bf Z} = a[{\bf X,Z}]+b[{\bf Y,Z}]\\
& [{\bf Z},a{\bf X}+b{\bf Y}]={\bf Z} \times (a{\bf X}+b{\bf Y})={\bf Z}\times a {\bf X}+{\bf Z} \times b{\bf Y}=a[{\bf Z,X}]+b[{\bf Z,Y}]

\end{align*}
$$


自反性：
$$
[{\bf X,X}]={\bf X} \times {\bf X}={\bf 0}
$$


雅可比等价：
$$
\begin{align*}
&[{\bf X},[{\bf Y,Z}]]+[{\bf Y},[{\bf Z,X}]]+[{\bf Z},[{\bf X,Y}]]\\
& ={\bf X} \times {\bf Y} \times {\bf Z}+{\bf Y} \times {\bf Z} \times {\bf X}+{\bf Z} \times {\bf X} \times {\bf Y}\\
& = {\bf X}^{\wedge} {\bf Z}^{\wedge} {\bf Y}+{\bf Y}^{\wedge} {\bf X}^{\wedge} {\bf Z}+{\bf Z}^{\wedge} {\bf Y}^{\wedge} {\bf X}\\
& = 
\begin{bmatrix}
0 & -c_{1} & b_{1}\\
c_{1} & 0 & -a_{1}\\
-b_{1} & a_{1} &0
\end{bmatrix}
\begin{bmatrix}
0 & -c_{3} & b_{3}\\
c_{3} & 0 & -a_{3}\\
-b_{3} & a_{3} & 0
\end{bmatrix}
\begin{bmatrix}
a_{2}\\
b_{2}\\
c_{2}
\end{bmatrix}
+
\begin{bmatrix}
0 & -c_{2} & b_{2}\\
c_{2} & 0 & -a_{2}\\
-b_{2} & a_{2} &0
\end{bmatrix}
\begin{bmatrix}
0 & -c_{1} & b_{1}\\
c_{1} & 0 & -a_{1}\\
-b_{1} & a_{1} & 0
\end{bmatrix}
\begin{bmatrix}
a_{3}\\
b_{3}\\
c_{3}
\end{bmatrix}
+
\begin{bmatrix}
0 & -c_{3} & b_{3}\\
c_{3} & 0 & -a_{3}\\
-b_{3} & a_{3} &0
\end{bmatrix}
\begin{bmatrix}
0 & -c_{2} & b_{2}\\
c_{2} & 0 & -a_{2}\\
-b_{2} & a_{2} & 0
\end{bmatrix}
\begin{bmatrix}
a_{1}\\
b_{1}\\
c_{1}
\end{bmatrix}
\\
& = 
\begin{bmatrix}
a_{3}(b_{1}b_{2}+c_{1}c_{2})-a_{2}(b_{1}b_{3}+c_{1}c_{3})\\
b_{3}(a_{1}a_{2}+c_{1}c_{2})-b_{2}(a_{1}a_{3}+c_{1}c_{3})\\
c_{3}(a_{1}a_{2}+b_{1}b_{2})-c_{2}(b_{1}b_{3}+a_{1}a_{3})
\end{bmatrix}
+
\begin{bmatrix}
a_{1}(b_{2}b_{3}+c_{2}c_{3})-a_{3}(b_{1}b_{2}+c_{1}c_{2})\\
b_{1}(a_{2}a_{3}+c_{2}c_{3})-b_{3}(a_{1}a_{2}+c_{1}c_{2})\\
c_{1}(a_{2}a_{3}+b_{2}b_{3})-c_{3}(b_{1}b_{2}+a_{1}a_{2})
\end{bmatrix}
+
\begin{bmatrix}
a_{2}(b_{1}b_{3}+c_{1}c_{3})-a_{1}(b_{2}b_{3}+c_{1}c_{3})\\
b_{2}(a_{1}a_{3}+c_{1}c_{3})-b_{1}(a_{2}a_{3}+c_{1}c_{3})\\
c_{2}(a_{1}a_{3}+b_{1}b_{3})-c_{1}(b_{2}b_{3}+a_{1}a_{3})
\end{bmatrix}
\\
& = 
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
={\bf 0}
\end{align*}
$$


则$\mathfrak g =(\mathbb R^{3}, \mathbb R, \times)$构成李代数。



### 推导$SE(3)$的指数映射

推导$SE(3)$指数映射部分左雅可比的形式：
$$
\begin{align*}
J & \triangleq \sum_{n=0}^{\infty} \frac{1}{(n+1)!}(\phi ^{\wedge})^{n}=\sum_{n=0}^{\infty} \frac{1}{(n+1)!}(\theta a^{\wedge})^{n}\\
  & =I+\frac{1}{2!}\theta a^{\wedge}+\frac{1}{3!} \theta^{2}a^{\wedge}a^{\wedge}-\frac{1}{4!}\theta^{3}a^{\wedge}a^{\wedge}a^{\wedge}+\frac{1}{5!}\theta^{4}a^{\wedge}a^{\wedge}a^{\wedge}a^{\wedge}+\cdots\\
  & =aa^{T}-a^{\wedge}a^{\wedge}+\frac{1}{2!}\theta a^{\wedge}+\frac{1}{3!}\theta^{2}a^{\wedge}a^{\wedge}-\frac{1}{4!}\theta^{3}a^{\wedge}-\frac{1}{5!}\theta^{4}a^{\wedge}a^{\wedge}+\cdots \\
  & = \frac{1}{\theta}\left \{ aa^{T}\theta-a^{\wedge}a^{\wedge}\theta+a^{\wedge}-a^{\wedge}+\frac{1}{2!}\theta^{2}a^{\wedge}+\frac{1}{3!}\theta^{3}a^{\wedge}a^{\wedge}-\frac{1}{4!}\theta^{4}a^{\wedge}-\frac{1}{5!}\theta^{5}a^{\wedge}a^{\wedge}+\cdots \right \} \\
  & = \frac{1}{\theta} \left \{ aa^{T}\theta - a^{\wedge}a^{\wedge}(\theta-\frac{1}{3!}\theta^{3}+\frac{1}{5!}\theta^{5}+\cdots)-a^{\wedge}(1-\frac{1}{2!}\theta^{2}+\frac{1}{4!}\theta^{4}+\cdots)+a^{\wedge}  \right\}\\
  & = \frac{1}{\theta} \left \{ aa^{T}\theta - a^{\wedge}a^{\wedge}\sin \theta -a^{\wedge}\cos \theta +a^{\wedge}     \right \}\\
  & = \frac{1}{\theta} \left \{ aa^{T}\theta-\sin \theta (aa^{T}-I)+a^{\wedge}(1-\cos \theta)   \right  \} \\
  & = \frac{1}{\theta} \left \{  \sin \theta I +(\theta-\sin \theta)aa^{T}+(1-\cos \theta)a^{\wedge}   \right \} \\
  & = \frac{\sin \theta}{\theta}I + (1-\frac{\sin \theta}{\theta})aa^{T}+\frac{1-\cos \theta}{\theta}a^{\wedge}
\end{align*}
$$
至于为什么雅可比矩阵是这种形式，即$J=\sum_{n=0}^{\infty} \frac{1}{(n+1)!}(\phi ^{\wedge})^{n}$是从何而来的，这个在` state estimation for robotics`一书的223-226有讲到，但是写的很抽象，下面给出证明：
$$
\begin{align*}
& \xi ^{\wedge} = 
\begin{bmatrix}
\phi ^{\wedge} & \rho \\
{\bf 0}^{T} & 0
\end{bmatrix}
\\
& \xi ^{\wedge} \xi^{\wedge}= 
\begin{bmatrix}
\phi ^{\wedge} & \rho \\
{\bf 0}^{T} & 0
\end{bmatrix}
\begin{bmatrix}
\phi ^{\wedge} & \rho \\
{\bf 0}^{T} & 0
\end{bmatrix}
=\phi ^{\wedge} \xi^{\wedge}
\\
& \xi^{\wedge}\xi^{\wedge}\xi^{\wedge}=(\phi ^{\wedge})^{2}\xi^{\wedge}
\end{align*}
$$

$$
\begin{align*}
\exp(\xi ^{\wedge}) & = \sum_{n=0}^{\infty}\frac{1}{n!}(\xi ^{\wedge})^{n}\\
                    & = I+\xi^{\wedge}+\frac{1}{2!}\xi^{\wedge}\xi^{\wedge}+\frac{1}{3!}\xi^{\wedge}\xi^{\wedge}\xi^{\wedge}+\cdots \\
                    & =I+\xi^{\wedge}+\frac{1}{2!}\phi^{\wedge}\xi^{\wedge}+\frac{1}{3!}(\phi^{\wedge})^{2}\xi^{\wedge}+\cdots \\
                    & = I+\sum_{n=0}^{\infty}\frac{1}{(n+1)!}(\phi^{\wedge})^{n}\xi^{\wedge}\\
                    & = 
                    \begin{bmatrix}
                    I+\sum_{n=0}^{\infty}\frac{1}{(n+1)!}(\phi^{\wedge})^{n+1} & \sum_{n=0}^{\infty}\frac{1}{(n+1)!}(\phi^{\wedge})^{n}\rho \\
                    {\bf 0}^{T} & 1
                    \end{bmatrix}
                    \\
                    & = 
                    \begin{bmatrix}
                    \sum_{n=0}^{\infty}\frac{1}{n!}(\phi^{\wedge})^{n} & \sum_{n=0}^{\infty}\frac{1}{(n+1)!}(\phi^{\wedge})^{n}\rho \\
                    {\bf 0}^{T} & 1
                    \end{bmatrix}

\end{align*}
$$



### 伴随

先证明$\forall a \in \mathbb R^{3}, Ra^{\wedge}R^{T}=(Ra)^{\wedge}$,高翔提供的网址的证明不严谨，即
$$
(Ra)^{\wedge}v=(Ra)\times v =(Ra) \times (RR^{-1}v)=R(a \times R^{-1}v)=Ra^{\wedge}R^{-1}v
$$

$$
AB=BC,不能推导出A \neq B \quad\text{A,B,C是矩阵}
$$

下面通过旋转矩阵正交的性质来证明该式成立。
$$
\begin{align*}
& 设旋转矩阵R=
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\quad \text{这里的${\bf r}_{1},{\bf r}_{2},{\bf r}_{3}$既可以看作是$1 \times 3$的单位向量，也可以看作是矩阵，因此下面不再进行区分}\\
& 对于Rp^{\wedge}R^{T} = (Rp)^{\wedge} \iff p^{\wedge}=R^{T}(Rp)^{\wedge}R，因此转为证明后式\\
\end{align*}
$$

 在证明之前，有两个事项需要注意：

1.${\bf r}_{i}^{T} {\bf r}_{j}={\bf r}_{i} \cdot {\bf r}_{j}$  前面可以看作矩阵，后面看作向量点乘，${\bf r}$是列向量

2.因为$R$是旋转矩阵，因此行向量和列向量都是单位向量且两两正交。为了后面叉乘运算的一致性，需要将${\bf r}_{1},{\bf r}_{2},{\bf r}_{3}$分别看成三维正交坐标系的$x,y,z$轴，即：
$$
\begin{cases}
{\bf r}_{1} \times {\bf r}_{2} = {\bf r}_{1}^{\wedge}{\bf r}_{2}={\bf r}_{3}\\
{\bf r}_{2} \times {\bf r}_{3} = {\bf r}_{2}^{\wedge}{\bf r}_{3}={\bf r}_{1}\\
{\bf r}_{3} \times {\bf r}_{1} = {\bf r}_{3}^{\wedge}{\bf r}_{1}={\bf r}_{2}
\end{cases}
$$

$$
\begin{align*}
p^{\wedge} 
& = R^{T}(Rp)^{\wedge}R \\
& = 
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
(\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\begin{bmatrix}
p_{1}\\
p_{2}\\
p_{3}
\end{bmatrix})^{\wedge}
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\\
& =
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
({\bf r}_{1}p_{1}+{\bf r}_{2}p_{2}+{\bf r}_{3}p_{3})^{\wedge}
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\\
& = 
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
({\bf r}_{1}^{\wedge}p_{1}+{\bf r}_{2}^{\wedge}p_{2}+{\bf r}_{3}^{\wedge}p_{3})
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\\
& = 
p_{1}
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
{\bf r}_{1}^{\wedge}
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
+
p_{2}
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
{\bf r}_{2}^{\wedge}
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
+
p_{3}
\begin{bmatrix}
{\bf r}_{1}^{T}\\
{\bf r}_{2}^{T}\\
{\bf r}_{3}^{T}
\end{bmatrix}
{\bf r}_{3}^{\wedge}
\begin{bmatrix}
{\bf r}_{1} & {\bf r}_{2} & {\bf r}_{3}
\end{bmatrix}
\\
& = 
p_{1}
\begin{bmatrix}
{\bf r}_{1}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{1} & {\bf r}_{1}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{2} & {\bf r}_{1}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{3}\\
{\bf r}_{2}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{1} & {\bf r}_{2}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{2} & {\bf r}_{2}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{3}\\
{\bf r}_{3}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{1} & {\bf r}_{3}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{2} & {\bf r}_{3}^{T} {\bf r}_{1}^{\wedge} {\bf r}_{3}\\
\end{bmatrix}
+
p_{2}
\begin{bmatrix}
{\bf r}_{1}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{1} & {\bf r}_{1}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{2} & {\bf r}_{1}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{3}\\
{\bf r}_{2}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{1} & {\bf r}_{2}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{2} & {\bf r}_{2}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{3}\\
{\bf r}_{3}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{1} & {\bf r}_{3}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{2} & {\bf r}_{3}^{T} {\bf r}_{2}^{\wedge} {\bf r}_{3}\\
\end{bmatrix}
+
p_{3}
\begin{bmatrix}
{\bf r}_{1}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{1} & {\bf r}_{1}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{2} & {\bf r}_{1}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{3}\\
{\bf r}_{2}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{1} & {\bf r}_{2}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{2} & {\bf r}_{2}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{3}\\
{\bf r}_{3}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{1} & {\bf r}_{3}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{2} & {\bf r}_{3}^{T} {\bf r}_{3}^{\wedge} {\bf r}_{3}\\
\end{bmatrix}
\\
& = 
p_{1}
\begin{bmatrix}
0 & 0 & 0\\
0 & 0 & -1\\
0 & 1 & 0
\end{bmatrix}
+
p_{2}
\begin{bmatrix}
0 & 0 & 1\\
0 & 0 & 0\\
-1 & 0 & 0
\end{bmatrix}
+
p_{3}
\begin{bmatrix}
0 & -1 & 0\\
1 & 0 & 0\\
0 & 0 & 0
\end{bmatrix}
\\
& =
\begin{bmatrix}
0 & -p_{3} & p_{2}\\
p_{3} & 0 & -p_{1}\\
-p_{2} & p_{1} & 0
\end{bmatrix}
\end{align*}
$$

对于$SO(3)$上的伴随的证明只需进行泰勒展开即可，与上面的证明相同。
$$
\begin{align*}
& \exp ((Ad(R)P)^{\wedge}) = \exp ((Rp)^{\wedge})=\exp (Rp^{\wedge}R^{T})=\exp (R\theta a^{\wedge}R^{T})=\sum _{n=0}^{\infty}\frac{1}{n!}(R\theta a^{\wedge}R^{T})^{n} \\
& 令p=\theta a,a是单位向量，\theta为模长\\
& Ra^{\wedge}R^{T} Ra^{\wedge}R^{T} =Ra^{\wedge}a^{\wedge}R^{T}=R(aa^{T}-I)R^{T}=Raa^{T}R^{T}-I\\
& Ra^{\wedge}R^{T} Ra^{\wedge}R^{T} Ra^{\wedge}R^{T} =Ra^{\wedge}a^{\wedge}a^{\wedge}R^{T}=-Ra^{\wedge}R^{T}
\end{align*}
$$
泰勒展开后的式子利用正余弦函数表示，结果为：
$$
\begin{align*}
\exp(Rp^{\wedge}R^{T}) & = \sum _{n=0}^{\infty}\frac{1}{n!}(\theta R a^{\wedge}R^{T})^{n}\\
                       & = (1-\cos \theta)Raa^{T}R^{T}+\cos \theta I+\sin \theta Ra^{\wedge}R^{T}\\
                       & =R((1-\cos \theta)aa^{T}+\cos \theta I+\sin \theta a^{\wedge})R^{T}\\
                       & =R \exp(p^{\wedge})R^{T}
\end{align*}
$$
对于$SE(3)$的伴随$Ad(T)$有：
$$
\begin{align*}
& T\exp (\xi ^{\wedge})T^{-1} = \exp((Ad(T)\xi )^{\wedge})\\
& Ad(T)=
\begin{bmatrix}
R & t^{\wedge}R\\
{\bf 0} & R
\end{bmatrix}
\end{align*}
$$
$SO(3)$和$SE(3)$的伴随将在后面的位姿图优化中用到。



### 轨迹的描绘

1.$T_{WC}$是相机坐标系到世界坐标系的变换矩阵，其平移部分就是相机的移动距离，我们在解算位姿的时候是计算两帧之间的位姿，因此平移部分连起来就是相机的轨迹，即机器人的轨迹。

实际上，$T_{WC}$和$T_{CW}$之间只差了一个逆而已，都可以用来表示相机的位姿，但是实践当中使用$T_{CW}$更为常见，不过$T_{WC}$更为直观，因为$p_{W} =T_{WC}p_{C}=Rp_{C}+t_{WC}$对于相机原点来说，$p_{W}$就是在其对应于世界坐标系的点，而且正是$T_{WC}$的平移部分，那么连起来就可以看到相机的平移轨迹。

2.仿照ORB-SLAM2里的` CMakeLists.txt`写的CMakeLists.txt

```
cmake_minimum_required(VERSION 2.8)
project(trajectorydrawing)

set(CMAKE_BUILD_TYPE "Release")

# check C++11 or C++0x support
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${EIGEN3_SOURCE_DIR}
    ${Pangolin_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
)


add_executable(trajectorydrawing draw_trajectory.cpp)
target_link_libraries(trajectorydrawing ${EIGEN3_LIBRARIES} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})
```



读取数据的代码如下：

```
// 第一种方法，用fstream的getline分行读取

    ifstream fin(trajectory_file); //从文件中读取数据
    if(!fin.is_open()){
        cout<<"No "<<trajectory_file<<endl;
        return 0;
    }
    double t,tx,ty,tz,qx,qy,qz,qw;
    string line;
    while(getline(fin,line))
    {
       istringstream record(line); //从string读取数据
       record>>t>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
       Eigen::Vector3d p(tx,ty,tz);
       Eigen::Quaterniond q = Eigen::Quaterniond(qw,qx,qy,qz).normalized();
       Sophus::SE3 SE3_qp(q,p);
       poses.push_back(SE3_qp);
    }

    //第二种方法

    // ifstream in(trajectory_file);//创建输入流

    // if(!in){
    //     cout<<"open posefile failture!!!"<<endl;
    //     return 0;
    // }

    // for(int i=0; i<620; i++){

    //     double data[8]={0};
    //     for(auto& d:data) in>>d;//按行依次去除数组中的值

    //     Eigen::Quaterniond q(data[7], data[8], data[5], data[6]);
    //     Eigen::Vector3d t(data[1], data[2], data[3]);
    //     Sophus::SE3 SE3(q,t);
    //     poses.push_back(SE3);

    // }
    // end your code here

```



生成的轨迹图如下：

![trajectory](visual-SLAM-by-Gaoxiang-3\trajectory.png)



### 轨迹的误差

CMakeLists.txt文件内容如下：

```
cmake_minimum_required(VERSION 2.8)
project(trajectory_error)

set(CMAKE_BUILD_TYPE "Release")

# check C++11 or C++0x support
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${EIGEN3_SOURCE_DIR}
    ${Pangolin_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
)


add_executable(trajectory_error trajectory_error.cpp)
target_link_libraries(trajectory_error ${EIGEN3_LIBRARIES} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES})
```



trajectory_error.cpp文件内容如下：

```
#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
 
using namespace std;
using namespace Eigen;
 
void ReadData(string FileName ,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &poses);
double ErrorTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
        vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e);
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
        vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e);
 
int main(int argc, char **argv)
{
    string GroundFile = "./groundtruth.txt";
    string ErrorFile = "./estimated.txt";
    double trajectory_error_RMSE = 0;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g;
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e;
 
    ReadData(GroundFile,poses_g);
    ReadData(ErrorFile,poses_e);
    trajectory_error_RMSE = ErrorTrajectory(poses_g, poses_e);
    cout<<"trajectory_error_RMSE = "<< trajectory_error_RMSE<<endl;
    DrawTrajectory(poses_g,poses_e);
 
}
 
 
/***************************读取文件的数据，并存储到vector类型的pose中**************************************/
void ReadData(string FileName ,vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> &poses)
{
    ifstream fin(FileName);  //从文件中读取数据
   //这句话一定要加上，保证能够正确读取文件。如果没有正确读取，结果显示-nan
    if(!fin.is_open()){
        cout<<"No "<<FileName<<endl;
        return;
    }
    double t,tx,ty,tz,qx,qy,qz,qw;
    string line;
    while(getline(fin,line)) 
    {
        istringstream record(line);    //从string读取数据
        record >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Vector3d p(tx, ty, tz);
        Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();  //四元数的顺序要注意
        Sophus::SE3 SE3_qp(q, p);
        poses.push_back(SE3_qp);
    }
 
}
 
/*******************************计算轨迹误差*********************************************/
double ErrorTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
        vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e )
{
    double RMSE = 0;
    Matrix<double ,6,1> se3;
    vector<double> error;
    for(int i=0;i<poses_g.size();i++){
        se3=(poses_g[i].inverse()*poses_e[i]).log();  //这里的se3为向量形式，求log之后是向量形式
        //cout<<se3.transpose()<<endl;
        error.push_back( se3.squaredNorm() );  //二范数
       // cout<<error[i]<<endl;
    }
 
    for(int i=0; i<poses_g.size();i++){
        RMSE += error[i];
    }
    RMSE /= double(error.size());
    RMSE = sqrt(RMSE);
    return RMSE;
}
/*****************************绘制轨迹*******************************************/
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_g,
        vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses_e) {
    if (poses_g.empty() || poses_e.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }
 
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);  //创建一个窗口
    glEnable(GL_DEPTH_TEST);   //启动深度测试
    glEnable(GL_BLEND);       //启动混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//混合函数glBlendFunc( GLenum sfactor , GLenum dfactor );sfactor 源混合因子dfactor 目标混合因子
 
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0) //对应的是gluLookAt,摄像机位置,参考点位置,up vector(上向量)
    );
 
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
 
 
    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
 
        glLineWidth(2);
        for (size_t i = 0; i < poses_g.size() - 1; i++) {
            glColor3f(1 - (float) i / poses_g.size(), 0.0f, (float) i / poses_g.size());
            glBegin(GL_LINES);
            auto p1 = poses_g[i], p2 = poses_g[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
 
        for (size_t j = 0; j < poses_e.size() - 1; j++) {
            //glColor3f(1 - (float) j / poses_e.size(), 0.0f, (float) j / poses_e.size());
            glColor3f(1.0f, 1.0f, 0.f);//为了区分第二条轨迹，用不同的颜色代替,黄色
            glBegin(GL_LINES);
            auto p1 = poses_e[j], p2 = poses_e[j + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
 
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
 
}
```



![error](visual-SLAM-by-Gaoxiang-3\trajectory_error.png)











