---
title: A Brief Review of Object Detection and Semantic Segmentation
mathjax: true
tags:
  - computer vision
  - deep learning
  - object detection
  - semantic segmentation
categories: 科研记录
abbrlink: 152382790
date: 2019-02-27 14:58:38
---

​                          

## About

目标检测（object detection）和语义分割（semantic segmentation）是计算机视觉的两个重要研究内容，在人脸检测，视频监控和自动驾驶中都有很多的应用，它们对于机器理解环境也具有一定的作用。在视觉SLAM上，已经有科研工作者将这些研究成果加入到现有的SLAM系统中，以提高系统的鲁棒性，同时促进语义SLAM的发展。因此，需要对这些工作进行一些调研总结和学习理解。由于对深度学习不了解，还未系统地进行学习，所以写的内容可能幼稚了些，后续会再进行更新。

值得注意的是，目标检测在经典计算机视觉中也已经提出了一些特征描述的方法，在此不涉及，只涉及基于深度学习的目标检测方法，毕竟神经网络在图片特征的学习上比较擅长。

本博客将基于文末给出的参考文献来进行总结和学习，尽力把这些问题和技术梳理清楚，同时加入一些自己的思考和不成熟的想法。

![focus on method based on deep learning](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\focus_of_object_detection_based_on_DL.PNG)

## Task Definition

1. image classification--multi level  图像分类，这个任务指的是给出一张图片，要识别出哪张图片属于哪个类别；
3. object detection--what categories and location  物体检测，这个任务除了需要识别出图片属于哪个类别，还需要对相应的物体进行具体位置的定位，我们通常用矩形框来框出这个物体；
4. semantic segmentation--pixel wise, but does not distinct everything in one category   语义分割，这个任务是指对图片中的每个 pixel 打上标签，比如这里要给它们打上 person、sheep、dog 的标签，需要进行非常精细的分类；
5. instance segmentation--object detection+semantic segmentation   实例分割，可以理解为进行物体检测后，对每个矩形框中的物体进行语义分割，该任务除了需要找到物体的类别和位置之外，还需要分辨出不同物体的 pixel；


![task definition](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\task definition.png)

<!--more-->

## Object Detection

### Progress (一步法<font color=DeepSkyBlue>single-stage</font>, 两步法two stage) 

最近有一篇很好的综述[[5]](https://arxiv.org/pdf/1809.02165v2.pdf)详细地梳理了最近5年来的generic object detection技术发展和比较的综述。

![object detection developing route](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\developing_route_of_object_detection.jpeg)

![progress](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\recent progress in object detection.png)

上图绿色的字体表示的是 Two-stage Detector 的发展历程，当然还有很多其他的方法，这里列出的是一些比较有代表性的方法。

- 2014 年有一项很重要的工作是 R-CNN，它是将物体检测首次应用于深度学习中的一篇论文，它的主要思路是将物体检测转化为这么一个问题：首先找到一个 region（区域），然后对 region 做分类。之后作者又提出了 Fast R-CNN，它是一个基于 R-CNN 的算法，运算速度显著提高。

- 2015 年，这一群人又提出了 Faster R-CNN，它在速度上相比 Fast R-CNN 有了更大的提高，主要是改进了怎样在 Fast R-CNN 和 R-CNN 中找 region 的过程，Faster R-CNN 也是用深度学习的方法得到一些 region（称之为 proposal），然后再用这些 proposal 来做分类。虽然距离 Faster R-CNN 的提出已经三年多了，但它依旧是使用非常广泛的一种算法。

- 2016 年，代季峰等人提出了 R-FCN，它在 Faster R-CNN 的基础上进行了改进，当时它在性能和速度都有非常大的提高。

- 2017 年有两篇影响力非常大的论文，FPN 和 Mask R-CNN。FPN 也就是 Feature Pyramid Network，何恺明大神的论文，它相当于生成了 feature pyramid，然后再用多个 level 的 feature 来做 prediction。Mask R-CNN 这篇论文获得了 ICCV 2017 的最佳论文，也是何恺明大神的作品，其在 Faster R-CNN 基础上增加了 mask branch，可以用来做实例分割，同时因为有 multi-task learning，因此它对物体框的性能也有很大的提高。（另外，今年年初，2019.1月，何恺明将自己的Mask R-CNN和FPN合在了一起，效果也不错）。

- 2018 年，沿着 Faster R-CNN 这条路线提出的方法有 Cascade R-CNN，它将 cascade 结构用在了 Faster R-CNN 中，同时也解决了一些 training distribution 的一些问题，因此它的性能是比较高的。另外还有两篇比较重要的论文——Relaiton Network 和 SNIP。

上图蓝色字体表示的 Single-stage Detector 的发展历程：2014 年的 MultiBox 是非常早期的工作；2016 年提出了 SSD 和 YOLO，这两篇论文是 Single-stage Detector 中的代表作；2017 年提出了 RetinaNet（当时 Single-stage Detector 中性能最高的方法）和 YOLO v2；2018 年有一个新的思路，提出了 CornerNet，把物体检测看成一堆点的检测，然后将这些点 group 起来。



### General Pipeline of Object Detection

参考文献[[2]](https://www.cnblogs.com/skyfsm/p/6806246.html)很好地解释了深度学习进行object detection的思路是什么。

如前所述，基于深度学习的Object Detection的方法主要有两条发展脉络，一个是**single-stage detector**，另一个是**two-stage detector**.



**Two-stage detector**

![single-stage detector](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\two-stage detector.png)

- 第一部分是 Feature generation：首先图片经过 backbone（分类网络）后，会生成 feature； 之后 feature 或者直接进行 prediction，或者再过一个 neck 进行修改或增强；

- 第二部分是 Region proposal：**这个部分是 Two-stage Detector 的第一个 stage**，其中会有一堆 sliding window anchor（预先定义好大小的框），之后对这些框做 dense 的分类和回归；接着再筛除大部分 negative anchor，留出可能是物体的框，这些框称之为 proposal；

- 第三个部分是 Region recognition：有了 proposal 后，在 feature map 上利用 Rol feature extractor 来提取出每一个 proposal 对应的 feature（Rol feature），最后会经过 task head；

  

**Single-stage detector**

![single-stage detector](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\single-stage detector.png)

- 图片首先经过 feature generation 模块，这里也有 sliding window anchor，但是它们不是用来生成 proposal 的，而是直接用于做最终的 prediction，通过 dense 的分类和回归，能直接生成最终的检测结果。



**一步法和两步法的区别到底在哪里？**

实际上，现在的二步法object detection的框架都是基于Faster R-CNN的，一步法中的比较全面的算法是SSD，这两个在后面我都会进行详细地阐述。

![comparison of one stage and two stage](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\one_two_stage_comparison.PNG)

对于两步法的基础Faster R-CNN，两个阶段分别在RPN和R-CNN中完成，第一阶段是预设一系列不同大小和比例的anchors，然后将整张图传入CNN提取特征，最后利用PRN对anchors进行分类和回归，得到候选区域（proposals）；第二阶段是利用RolPooling扣取每个候选区域的特征，接着把扣取特征的特征送入后续R-CNN网络，最后对候选区域进一步分类和回归，得到最终的检测结果。

由上图也可以看出，两步法是相对于一步法多了二阶段的分类，回归和特征，因而精度会更好，但是会使得算法运行的时间加长。也就是说，**一步法也就是类似于只有RPN阶段，而二步法多了R-CNN**，使得分类更加精细，同时再对候选区域进行回归。

**那么一步法是否可以仿照这样的思路或者利用其他的方法达到二步法的精度，却又不损失运行效率？**

在参考文献[[3]](https://mp.weixin.qq.com/s/84JG1ZGFKb6Xp3WQFjtxZw)里面，张士峰博士分享了他们的工作[Single-Shot Refinement Neural Network for Object Detection](<https://github.com/sfzhang15/RefineDet>)，在SSD中加入一些类似R-CNN作用的模块，来提高检测的精度，同时也保持了原有的运行效率。

![RefineDet](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\architecture_of_refinedet.PNG)

整个结构包含ARM模块，TCB连接和ODM模块，其中ARM模块和ODM模块分别对应二步法中的RPN和R-CNN的功能，TCB的主要作用是转换ARM特征，融合高层特征，因为两步法中的特征不相同，第二阶段会提取新的特征，因此需要转换和融合。**该方法与二步法的区别在于没有RoIPooling**，因此运行时间不会边长。该方法的精度大概较SSD提升了两个点，速度上会更快些，因为输入图像的分辨率可以是320x320的。具体的检测框架和测试结果可以看[原论文](<https://arxiv.org/pdf/1711.06897v3.pdf>)。

---

借此提下object detection近年来的发展趋势应当不再是该网络调参来刷速度和正确率，而是回归到框架本身的反思和设计上，毕竟现有的算法都是从Faster R-CNN来的，因此是否可以反思该框架的问题，然后对其进行改善，或者直接提出新的检测思路，比如说在前处理阶段的anchors和后处理阶段的NMS都是手动设置的，这个是否可以进行自动调节，也是一个值得研究的点。

另外，在视觉SLAM和很多其他的应用场景上，输入的都是视频流的形式，而不是单个的图像，这一方面对算法的实时性，硬件的运算速度提出了要求，另一方面也对视频物体检测提出了要求，如何利用帧与帧之间的信息来加速物体检测，也是有待研究的。

最后一点就是，视觉SLAM的最终梦想是能在现实大环境下进行运行，然后给出环境地图，以此与人进行交互。现实场景很复杂，充满动态的，不确定性的因素，因此这样的功能要想实现还有很长的路要走。不过可以预见的是，视觉SLAM需要结合这些新的，基于学习方法的计算机视觉的研究成果，以帮助机器人来理解环境，提高系统的鲁棒性，同时各个模块和任务之间也将是相互联系相互促进的，多任务联合（multi-task），以此提升系统的容错性 和运行时间。

---



### Faster R-CNN

Faster R-CNN 是 Two-stage Detector 工作的基础，它主要提出了两个东西:

- **RPN**：即 Region Proposal Network，目前是生成 proposal 的一个标准方式;

- **Traning pipeline**：主要讲的是 Two-stage Detector 应该怎样 train，论文里给出的是一种交替(alternating training)的方法;

![RPN](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\RPN.png)

![alternating training](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\alternating training.png)

![joint training](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\joint training.png)

因为深度学习的发展，大家都不想一步一步来回调参，因此会有了**joint training**这个东西。



### FPN

- FPN（Feature Pyramid Network）主要也是提出了两项重要的思路：**Top-down pathway 和 Multi-level prediction。**

- 下图中的 4 张图代表了基于单个或多个 feature map 来做预测的 4 种不同的套路：

![four ways of FPN](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\four ways of FPN.png)

- 具体实现如下图所示：

![pipeline of FPN](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\pipeline of FPN.png)



### MASK R-CNN

Mask R-CNN 主要也有两点贡献：**RoIAlign**和**Mask branch**

- RoIAlign：在 Mask R-CNN 之前，大家用得比较多的是 Rol Pooling，实现过程是：给出一个框，在 feature map 上 pool 出一个固定大小的 feature，比如要 pool 一个 2×2 的 feature，首先把这个框画出 2×2 的格子，每个格子映射到 feature map，看它覆盖了多少个点，之后对这些点做 max pooling，这样就能得出一个 2×2 的 feature。它的劣势是如果框稍微偏一点，得出的 feature 却可能是一样的，存在截断误差。RolAlign 就是为了解决这一问题提出的。Rol Align 并不是直接对框内的点做 max pooling，而是用双线性插值的方式得到 feature。其中还有一步是：在 2×2 的每个框中会选择多个点作为它的代表，这里选择了 4 个点，每个点分别做双线性插值，之后再让这 4 个点做 max/average pooling

- Mask branch：它是一个非常精细的操作，也有额外的监督信息，对整个框架的性能都有所提高。它的操作过程如下图所示:

![mask branch](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\mask branch.png)



### Cascade R-CNN

- Cascade R-CNN 是目前 Faster R-CNN 这条线中较新的方法。这个方法也提出了两点贡献：一是提出了使用 **cascade architecture**；二是提出了怎样来适应 **training distribution**。

- Cascade architecture ：这个框架不是特别新的东西，之前也有类似的结构。下图左边是 Faster R-CNN，右边是 Cascade R-CNN。其中 I 代表图像或者图像生成的 feature map，H0 是 RPN，B 是 bounding box regression，C 是 classification。经过 RPN 得到的 proposal 再做 pooling 后，会有分类和回归这两个 prediction。

![cascade architecture](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\cascade architecture.png)

- Cascade R-CNN 的结构是，在经过第一次分类和回归之后，会用得到 bounding box 再来做一次 pooling，然后对这些框做下一阶段的分类和回归，这个过程可以重复多次。但如果仅仅使用 Cascade R-CNN 而不做其他改变，Cascade R-CNN 带来的提高是非常有限的。 Cascade R-CNN 提出了一个很好的 motivation，这是它比较有意义的一个地方。它研究了一下采用不同的 IoU 阈值来进行 training 的情况下，Detector 和 Regressor 的性能分布。



### RetinaNet

- RetinaNet 是 Singe-stage Detector 目前比较重要的一项工作，它可以看做是由 FPN+Focal Loss 组成的，其中 FPN 只是该论文中用到的架构，而 Focal Loss 则是本论文主要提出的工作。

- RetinaNet 结构如下：

![RetinaNet](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\RetinaNet.png)

- RetinaNet 的结构 和 SSD 非常类似，只不过它用的是 ResNet，并在 ResNet 上加入了 FPN 的结构，每一层都有两个分支：一个用来做分类；另一个用来做框回归。此外它的每个 head 都比 SSD 和 Faster R-CNN 都要大一点，这样的话，它的参数量比较大，计算速度也比较慢。

- 而 Focal Loss 试图解决的问题是 class imbalance。针对 class imbalance 的问题，Two-stage Detector 一般是通过 proposal、mini-batch sampaling 两种方式来解决的；SSD 是通过 hard negative mining 来解决的；而 RetinaNet 则通过 Focal Loss 来解决该问题。

- Focal loss 的核心思路是：对于 high confidence 的样本，给一个小的 loss——这是因为正负样本不平衡，或者说是由于 class imbalance 导致了这样的问题：比如说正负样本的比例是 1:1000，虽然负样本的 loss 都很小，但数目非常多，这些负样本的 loss 加起来的话，还是比正样本要多——这样的话，负样本就会主导整个框架。



### Relation Network

- 在 Relation Network 之前的大部分 detectcor，在做 prediction 或 training 的时候通常只考虑到当前这一个框，而 Relation Network 提出还要考虑这个框周围的框，并基于此提出了一个 relation module，可以插在网络的任何位置，相当于是 feature refinement。Relation module 如下图所示：

![Relation Network](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\relation network.png)

- 它的核心思想是：当前框的 feature 除了由当前框决定之外，还要考虑当前框和周围框及其它框的关系



### SNIP

- SNIP（Scale Normalization for Image Pyramids）另一篇比较有启发性的工作。它提出的问题是：在 train 分类器的时候，究竟是要 scale specific 还是 scale invariant。传统的 detector 通常会选择 scale invariant，但 SNIP 研究了一下之前的方法后，发现之前的训练方式得到的 feature 对 scale 并没不是很 robust，因而提出要尽量减少 scale 的 variance，让训练时的 scale 尽可能地相似。

- SNIP 结构图如下:

![SNIP](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\SNIP.png)



### CornerNet

- CornerNet，是 Singe-stage Detector 中比较新的方法，其与其他方法最不一样的地方是：之前的方法会在图像上选出框，再对框做分类的问题；CornerNet 则是在图中找到 pair 的关键点，这个点就代表物体。它的 pipeline 包括两步：第一步是检测到这样的 corner，即 keypoint；第二步是 group corner，就是说怎样将同一个物体的左上顶点和右下顶点框到一起。

- 其算法结构如下：

![CornerNet](A-Brief-Review-of-Object-Detection-and-Semantic-Segmentation\cornerNet.png)

### [mmdetection](https://github.com/open-mmlab/mmdetection)

港中文多媒体实验室的开源物体检测框架。



## References

[1]. [物体检测算法的近期发展及开源框架介绍-陈恺](<http://www.mooc.ai/open/course/604>)
[2]. [基于深度学习的目标检测技术演进：R-CNN、Fast R-CNN、Faster R-CNN](https://www.cnblogs.com/skyfsm/p/6806246.html)
[3]. [基于深度学习的物体检测算法对比探索 - 张士峰](https://mp.weixin.qq.com/s/84JG1ZGFKb6Xp3WQFjtxZw)
[4]. [基于深度学习的目标检测算法近5年发展历史（综述）](<https://blog.csdn.net/Gentleman_Qin/article/details/84421435>)
[5]. [Deep Learning for Generic Object Detection: A Survey](https://arxiv.org/pdf/1809.02165v2.pdf)
[6]. [目标检测算法中检测框合并策略技术综述](<https://zhuanlan.zhihu.com/p/48169867>)
[7]. [综述 | CVPR2019目标检测方法进展](<https://mp.weixin.qq.com/s/mu_4kNGZuExxUK2JFTdDFw>)

