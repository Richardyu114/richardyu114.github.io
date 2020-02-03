---
title: FutureMapping2
mathjax: true
tags:
  - SLAM
  - mapping
  - AI system
  - computer vision
  - hardware
categories: 论文阅读
abbrlink: 147804227
date: 2019-11-17 16:24:29
---

## About

继第一篇[FutureMapping]( https://arxiv.org/abs/1803.11288 )之后，视觉SLAM领域内的奠基者Andrew Davison最近又将他的和别人讨论的有关未来空间AI对地图构建，机器人协同定位，以及动态问题等新想法撰写成了新的论文[FutureMapping2]( https://arxiv.org/abs/1910.14139 )，置顶在了自己的推特上，表示欢迎大家交流自己的想法。

![Andrew-twitter](FutureMapping2\Andrew-twitter.PNG)

总的来说，这篇文章干货还算是很多的，主要是着眼于factor graph（因子图）和Gaussian Belief Propagation（GBP，高斯置信传播）对整体，动态机器人建图等问题的潜力和前景，不仅在数学上进行一些tutorial，还给出了三个python demos。由于我对视觉SLAM只是了解整体特点，其中各种计算细节和优化方法并没有认真看过和代码书写过，因此读完这篇充满amazing reflections的文章之后我只能把握到整体的idea和一些实现方法。对于GBP和factor graph 结合构建的数学模型，我将在后续花时间弄懂后再对该blog进行补充。

**非常建议对SLAM或者机器视觉领域感兴趣的同行阅读此论文，相信会帮助您开阔思路！**

<!-- more -->

## Content

### Idea 

> Abstract:  We argue the case for Gaussian Belief Propagation (GBP) as a strong algorithmic framework for the distributed, generic and incremental probabilistic estimation we need in Spatial AI as we aim at high performance smart robots and devices which operate within the constraints of real products. Processor hardware is changing rapidly, and GBP has the right character to take advantage of highly distributed processing and storage while estimating global quantities, as well as great flexibility. We present a detailed tutorial on GBP, relating to the standard factor graph formulation used in robotics and computer vision, and give several simulation examples with code which demonstrate its properties.  

关键词：分布式，边缘计算，局部估计，GBP，因子图，概率模型

**（边缘计算，分布式的，局部计算和存储，整体计算和构建以一种‘graph’的方式展开）**

首先，背景是现在的spatial AI系统要试着处理异质的数据，通过不同的估计手段将其转换成一致性的表示方式，但是这受限于现在的处理器性能（实时，持续地处理带来计算负担，存储负担和转换负担）。Andrew认为目前有两种方式可以对此进行提升：

- One is to focus on scene representation, and to find new parameterisations of world models which allow high quality scene models to be built and maintained much more efficiently. 这是表征方式的问题，这也是学术界都在解决的问题，representation learning. 机器人，计算机如何利用自己的硬件特点来认识和表征周围的世界，这跟人认识世界不一定是一样的。
-  The other is to look towards the changing landscape in computing and sensing hardware. 另一个就是硬件设计，比如现在视觉SLAM中的“event camera”，利用事件来记录。这里Andrew也推荐了ETH苏黎世联邦理工大学 J. Martel的phD thesis，[传送门]( https://www.research-collection.ethz.ch/handle/20.500.11850/362900 )

硬件问题上，Andrew并没有说太多，大都还是一些常见的设想，在后续的section中，Andrew主要是是针对第一点来说的。

> The purest representation of the knowledge in a **Spatial AI problem is the factor graph itself, rather than probability distributions derived from it, which will always have to be stored with some approximation**. What we are really seeking is an algorithm which implements Spatial AI in a distributed way on a computational resource like a graph processor, by storing the factor graph as the master representation and operating on it in place using local computation and message passing to implement estimation of variables as needed but taking account of global influence. 

(有关因子图的资料，推荐[Factor Graphs for Robot Perception]( http://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf )， 关于高斯置信传播，Andrew是根据bishop的PRML一书来的，此外他在文中推荐的是比较老的参考文献[Factor Graphs and the Sum-Product Algorithm]( http://web.cs.iastate.edu/~honavar/factorgraphs.pdf )，我自己也在网上发现了一篇伯希来大学的phD论文[ Gaussian Belief Propagation: Theory and Application ]( https://arxiv.org/pdf/0811.2518.pdf )，希望对后续理解有所帮助。）

Andrew认为可以通过因子图来进行局部估计，进行边缘计算，然后进行信息传递，通过这种方法来构建分布式和全局上的计算，从而达到认知和执行任务的目的。

---to be continued

### Mathematics models

#### factor graph



#### GBP



### Examples







