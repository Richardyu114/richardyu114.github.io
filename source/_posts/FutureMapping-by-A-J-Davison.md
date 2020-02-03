---
title: FutureMapping by A.J.Davison
mathjax: true
tags:
  - SLAM
  - mapping
  - AI system
  - computer vision
  - hardware
categories: 论文阅读
abbrlink: 1703359370
date: 2019-04-16 16:24:31
---

## About

----

IML的[A.J.Davison](<https://www.imperial.ac.uk/people/a.davison>)教授是visual SLAM领域里的奠基人之一，近期抽空看了他在推特上置顶的有关SLAM未来的一些思考的论文：  **FutureMapping: The Computational Structure of Spatial AI Systems**。对于做SLAM的同学，感觉这篇论文还是值得一看的，这里有个博主将该论文翻成了[中文](<https://blog.csdn.net/cicibabe/article/details/79846466>)。

实际上，通篇读下来我的感受是并没有发现Davison提出了比较吸引人眼球的见解，不过有不少亮点，也让我加深了对SLAM这个东西的理解。虽然他对未来visual SLAM的功能性估计也和综述**Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age**  里描述得差不多，只不过后者说的比较“大而空”，都是一些常见的想法，什么动态啦，语义融合啦，模仿生物视觉啦，地图表征与更新等等，但是在Davison的这篇论文里，他以一个机器人学者的角度出发，试图从硬件和软件这两个方面去思考，未来的需求下机器人应该如何完成视觉任务，硬件应该如何发展去支持算法有效的计算，以及整个系统该有怎样的结构，才能使得机器人更好地在不同的场景下，甚至是大场景中完成不同的任务。

作为一个入门SLAM不算太久的工科学生来说，虽然不少技术知识还未掌握，但是偶尔看看这样的文章和思考也不是未尝不可，至少能激发我思考什么样的东西是值得做的，哪一个技术是未来需求的？

不过有一个疑问是一直存在我心中的，大家都谈到了未来的visual SLAM会结合机器学习或者模仿生物视觉机制和大脑存储记忆的机制，这个我是赞同的，**但是对于三维地图而言，是否是有必要去重建的**，除了它们在AR上的一些应用？大家都说想要融合语义标签，物体分类与识别融合到三维图里面，有的人可能还想让地图进行实时更新，容纳动态物体（至少我曾经是这样想的），但是这样的做的目的和意义到底是什么？单纯说目的是为了让机器进一步理解环境是无法让我满足的，所以我也试图在寻找和思考这个问题的答案。遗憾的是，Davison在这篇文章里面也没有提到此类问题，不过也没有夸大三维地图的作用，而是强调图模型的作用，这一点我是赞同的。也就是说，类似人一样，我们利用视觉和计算完成任务时，"graph"这个东西是肯定发挥了很大的作用，但是却没必要“事无巨细”的记下来，我们大多是提取重要的特征，压缩下来，然后进行推断，从而得出各种预测和结论，而且事后也可以在脑海中回忆重建出场景的三维模型，此外，这些压缩信息也会随时间及时进行更迭，经常重访的则会记得牢固一些，调取起来也很快，那些不常去的可能就会进一步压缩或者删除了。

Davison在论文是这样描述地图表征的利用形式的：

> In real-time, the system must maintain and update a world model, with geometric and semantic information, and estimate its position within that model, from primarily or only measurements from its on-board sensors.
>
> The system should provide a wide variety of taskuseful information about **‘what’ is ‘where’ in the scene**. Ideally, **it will provide a full semantic level model of the identities, positions, shapes and motion of all of the objects and other entities in the surroundings**.
>
>The representation of the world model will **be close to metric**, at least locally, to enable rapid reasoning about arbitrary predictions and measurements of interest to an AI or IA system.
>
> It will probably **retain a maximum quality representation of geometry and semantics only in a focused manner**; most obviously for the part of the scene currently observed and relevant for near-future interaction. **The rest of the model will be stored at a hierarchy of residual quality levels, which can be rapidly upgraded when revisited.**
>
>The system will be generally alert, in the sense that every incoming piece of visual data is checked against a forward predictive scene model: for tracking, and for detecting changes in the environment and independent motion. The system will be able to **respond to changes** in its environment.

所以我的意思是，在进行视觉任务时，重建三维地图应该不是必要的，至少在实际任务上目前可能起不到很大的作用，可能需要的是一种更加简洁凝练的图表征模式，这种模式更适合机器去认识环境，去进行编码，解码，计算，存储以及维护，而不是像人一样以为这样看到的是环境的理解方式，毕竟我们看到的是已经经过大脑处理的“人机交互结果”，并不是最核心的表征方式。但是大家为什么现在都比较热衷与做3D视觉，3D重建，我想可能是计算机视觉的一个难题吧，毕竟计算机图形学还是很有魅力的，毕竟未来的应用谁也说不准，只希望最后不要让这些技术让人类迷失在虚拟世界里。。。在这个方面，还得多一些[产品层面的思考](<https://www.technologyreview.com/s/613311/mapping-the-world-in-3d-will-let-us-paint-streets-with-augmented-reality/?utm_campaign=the_download.unpaid.engagement&utm_source=hs_email&utm_medium=email&utm_content=71836962&_hsenc=p2ANqtz-8xmRJCA-lRmkREUy6bMXIyqkH3NbAngJgynjDgBx2V-dGw-HLkRxMi3j1Z2izhsqPrpw6txfcuN5lVt9tU4FFzZ1ggiw&_hsmi=71836962>)。

以上只是我的一点不成熟的想法，还需要多去阅读思考和交流。

-------

<!-- more -->

## Content

这篇的文章的摘要如下：

> We discuss and predict the evolution of Simultaneous Localization and Mapping (SLAM) into a **general geometric and semantic ‘Spatial AI’ perception capability for intelligent embodied devices**. A big gap remains between the visual perception performance that devices such as augmented reality eyewear or consumer robots will require and what is possible within the constraints imposed by real products. Co-design of algorithms, processors and sensors will be needed. We **explore the computational structure of current and future Spatial AI algorithms and consider this within the landscape of ongoing hardware developments.**

可以看到Davison教授关注的是**general geometric and semantic ‘Spatial AI’ perception capability for intelligent embodied devices**。首先介绍了Spatial AI system的相关概念，`the goal of a Spatial AI system is not abstract scene understanding, but continuously to capture the right information, and to build the right representations, to enable real-time interpretation and action.` 然后分别从算法层面和硬件层面去探讨，什么样的元素应该具备，什么样的结构是合理的，什么样的计算方式和维护更新方式可能被采用，以及从被动的分析到主动的分析预测的思考（感觉这才是有点智能的味道），最后还批判现在的计算机视觉研究者都热衷于刷点，而不去思考架构本身的问题，这个吐槽很准了，毕竟听说今年CVPR2019刷点，刷速率的文章接受率都显著下降了，我们读者都疲劳了，更何况评委。当然教授最后吐槽的重点是为了思考Benchmark对visual SLAM的意义，因为visual SLAM是一个实践性很强的系统，是为了解决实际机器人问题而生的，因此在实际的实时实验中效果好才是真的好，一味地去比较各个指标没有太大的意义，而且也很难比较，指标又很多，比如最后文末列的：

> • Local pose accuracy in newly explored ares (visual odometry drift rate).
> • Long term metric pose repeatability in well mapped areas.
> • Tracking robustness percentage.
> • Relocalisation robustness percentage.
> • SLAM system latency.
> • Dense distance prediction accuracy at every pixel.
> • Object segmentation accuracy.
> • Object classification accuracy.
> • AR pixel registration accuracy.
> • Scene change detection accuracy.
> • Power usage.
> • Data movement (bits×millimetres).

文中的具体内容不再一一讲了，这里主要讲几个文章中让我感兴趣的点。

### ML 或者 DL能为Spatial AI system做什么

传统机器学习算法和深度学习中的神经网络擅长做分类和回归，它们在对图像的特征学习上有着得天独厚的优势。近些年也有好多工作是利用CNN和RNN，以及unsupervised learning等深度学习的方法来进行位姿估计和深度估计，也取得了不错的效果。不过有些人认为深度学习在已经研究差不多的3D Geometry上并没有什么意义，况且数学模型我们都知道，没必要去及蹭热度利用深度学习来做，相反那些现有的算法无法处理的图像问题，比如鲁棒性好的特征点提取，光照的变化，纹理的单一，场景的识别以及运动模糊等可以尝试利用深度学习隐式地解决。此外，深度学习在object detection，semantic segmentation等都有很好的成果，可以进行应用。

>These are learning architectures which use the designer’s knowledge of the structure of the underlying estimation problem to increase what can be gained from training data, and can be seen as hybrids between pure black box learning and hand-built estimation.     ...   Why should a neural network have to use some of its capacity to learn a well understood and modelled concept like 3D geometry? Instead it can focus its learning on the harder to capture factors such as correspondence of surfaces under varying lighting.
>
>These insights support our first hypothesis that future Spatial AI systems will have recognisable and general map-ping capability which builds a close to metric 3D model. This ‘SLAM’ element may either be embedded within a specially architected neural network, or be more explicitly separated from machine learning in another module which stores and updates map properties using standard estimation methods (as in SemanticFusion for instance). In both cases, there should be an identifiable region of memory which contains a representation of space with some recognisable geometric structure.



个人感觉机器学习以及现在，未来可能出现的一系列理论可能对Spatial AI system帮助最大地可能就是图像理解和环境表征方面了，另外长时间运行带来的认识融合和更新。以及压缩等，可能也会有帮助。有关这个方面的思考目前不是很深，因为我现在还没开始学习机器学习，所以对技术了解不深，但是我感觉这东西是个“万精油”，在SLAM上的应用很大程度上可能归功于设计者怎么用，用在哪里，也就是说怎么设计网络，然后通过什么去学习什么功能，而不是紧紧盯着传统的方法然后去想方设法实现它。



### 硬件与云端

硬件方面Davison教授主要探讨了装载该Spatial AI System的嵌入式硬件应该具有什么样的结构，而且还具有匹配视觉计算特点的算力，同时还得有一定的存储能力。他肯定了分割计算，并行计算，多核心，多线程，神经形态硬件架构的需求，也列出了一些正在研究的例子，具体的内容可以参见文章内容。

实际上，硬件对算法的促进具有着决定性的作用，从Yann Lecun在2019年的ISSCC上做的[报告](<https://pan.baidu.com/s/1lXv0aDSEKXKYQVhJc5X6-A>)就可以看出

，没有良好的硬件支持，算法根本没办法进行实验验证，就很难进步。因此，硬件方面的迫切需求是现在整个智能行业的燃眉之急。

![yan lecun lesson1 about hardware](FutureMapping-by-A-J-Davison\yanlecun_lesson1.jpeg)

另外，Davison教授也肯定了云端的重要性。因为云端相当于一个存储中心，可以存储环境表征这样的信息，而且可以同时对环境分布中的机器人进行通信和数据传输，这对机器人在大场景中，长时间执行任务起到“预热”等辅助性作用。

> Finally, when considering the evolution of the computing resources for Spatial AI, we should never forget that, cloud computing resources will continue to expand in capacity and reduce in cost. All future Spatial AI systems will likely be cloud-connected most of the time, and from their point of view the processing and memory resources of the cloud can be assumed to be close to infinite and free. What is not free is communication between an embedded device and the cloud, which can be expensive in power terms, particularly if high bandwidthdata such as video is transmitted. The other important consideration is the time delay, typically of significant fractions of a second, in sending data to the cloud for processing.
>
> The long term potential for cloud-connected Spatial AI is clearly enormous. The vision of Richard Newcombe, Director of Research Science at Oculus, is that all of these devices should communicate and collaborate to build and maintain shared a ‘machine perception map’ of the whole world. The master map will be stored in the cloud, and individual devices will interact with parts of it as needed. A
> shared map can be much more complete and detailed than that build by any individual device, due to both sensor coverage and the computing resources which can be put into it. A particularly interesting point is that the Spatial AI work which each individual device needs to do in this setup can in theory be much reduced. Having estimated its location within the global map, it would not need to densely map
> or semantically label its surrounding if other devices had already done that job and their maps could simply be projected into its field of view. It would only need to be alert for changes, and in turn play its part in returning updates.



### Graphs

Davison在论文的第5节讲了很多有关"Graphs"的东西，我们知道，现在的visual SLAM框架都开始逐渐认同将图优化作为减小估计误差的手段要比滤波器估计的效果好得多，因为"graphs"本身就是视觉的一种表征方式，而且在约束上具有非线性性，能更好地模拟现实情况。

在SLAM方面，教授主要提出了geometry和local appearance两者是否可以联系起来的观点：

> We have not yet discovered a **suitable feature representation which describes both local appearance and geometry in such a way that a relatively sparse feature set can provide a dense scene prediction.** We believe that learned features arising from ongoing geometric deep learning research will provide the path towards this.
>
> Some very promising recent work which we believe is heading in the right direction Bloesch et al.’s CodeSLAM. This method uses an image-conditioned autoencoder to discover an optimisable code with a small number of parameters which describes the dense depth map at a keyframe. **In SLAM, camera poses and these depth codes can be jointly optimised to estimate dense scene shape which is represented by relatively few parameters.** In this method, the scene geometryis stilllocked to keyframes, but we believe that the next step is to discover learned codes which can efficiently represent both appearance and 3D shape, and to make these the elements of a graph SLAM system.

Davison教授另外一个观点是该实时系统中的"Computation Graph"，并且再次提出了”object-oriented SLAM"的概念。

>  How can we get back to this ‘object-oriented SLAM’ capability in the much more general sense, where a wide range of object classes of varying style and details could be dealt with? As discussed before, **SLAM maps of the future will probably be represented as multi-scale graphs of learned features which describe geometry, appearance and semantics**. Some of these features will represent immediately recognised whole objects as in SLAM++. **Others will represent generic semantic elements or geometric parts (planes, corners, legs, lids?**) which are part of objects either already known or yet to be discovered. Others may approach surfels or other standard dense geometric elements in representing the geometry and appearance of pieces whose semantic identity is not yet known, or does not need to be known.
>
> **Recognition, and unsupervised learning, will operate on these feature maps to cluster, label and segment them**. The machine learning methods which do this job will themselves improve by selfsupervision during the SLAM process, taking advantage of dense SLAM’s properties as a "correspondence engine".

![some elements of computation graph](FutureMapping-by-A-J-Davison\some_elements_of_computation_graph.PNG)

这个图基本上等于是把系统算法的框架给列出来了，可以看出，核心还是”定位“（camera state）和”建图“（world model）。只不过里面加入了深度学习来提高系统的性能。

> **Most computation relates to the world model**, which is a persistent, continuously changing and **improving data store** where the system’s generative representation of the important elements of the scene is held; and the input camera data stream. Some of the main computational elements are:
>
> • Empirical labelling of images to features (e.g. via a CNN).
> • **Rendering**: getting a dense prediction from the world map to image space.
> • Tracking: aligning a prediction with new image data, including finding outliers and detecting independent movement.
> • **Fusion**: fusing updated geometry and labels back into the map.
> • **Map consolidation: fusing elements into objects, or imposing smoothing, regularisation**.
> • Relocalisation/loop closure detection: detecting self similarity in the map.
> • Map consistency optimization, for instance after confirming a loop closure.
> • **Self-supervised learning of correspondence information from the running system.**

这些都是当前比较主流的观点，而且里面涉及的知识体系比较庞大，因此大部分都是先针对一个来展开研究，不过我觉得要想对其进行突破，最大的，也是最有挑战性的问题应该就是世界模型表征问题了，对于机器来讲，这个应当是个非常简洁和高效的表征方式，同时也易于存储，调用，翻译和编码。



### 地图的处理，表示，预测和更新

其实这个部分前面已经提及了不少了，而Davison教授也单独在第6节讲了这个问题，对里面的几个关键问题进行了总结和思考：一个是硬件支持，一个是地图存储，一个是实时回环。

地图表征方面：

> There is a large degreeof choice possible in the representation of a 3D scene, but as explained in Section 5.1.2, we envision maps which **consist of graphs of learned features, which are linked in multi-scale patterns relating to camera motion**. These features must **represent geometry as well as appearance,** such that they can be used to render a dense predicted view of the scene from a novel viewpoint. It may be that they **do not need to represent full photometric appearance**, and that a somewhat abstracted view is sufficient as long as it captures geometric detail.

地图存储与维持方面（更新）：

> Within the main processor, a major area will be devoted to storing this map, in a manner which is **distributed around potentially a large number of individual cores which are strongly connected in a topology to mirror the map graph topology**. In SLAM, of course the map is defined and **grown dynamically**, so the graph within the processor must either be able to change dynamically as well, or must be initially defined with a large unused capacity which is filled as SLAM progresses.

>Importantly, a significant portion of the processing associated with large scale SLAM can be built directly into this graph. This is mainly the sort of ‘maintenance’ processing via which the map optimises and refines itself; including:
>
>• **Feature clustering; object segmentation and identification.**
>• Loop closure detection.
>• Loop closure optimization.
>• **Map regularisation (smoothing).**
>• **Unsupervised clustering to discover new semantic categories.**
>
>With time, data and processing, a map which starts off as dense geometry and low level features can be refined towards an efficient object level map. Some of these operations will run with very high parallelism, as each part of the map is refined on its local core(s), while other operations such as loop closure detection and optimisation will require message passing around large parts of the graph. Still, importantly, they can take place in a manner which is internal to the map store itself.

实时回环方面，教授提出了地图的存储与场景识别方面的一些难点，即“翻译”和“融合”之间协作的问题。因为相机的运动会对地图进行实时更新，该模块的重心在于维持，而不是比较数据，因此可能会对场景识别造成一定的影响。教授在这里提出了可以利用节点（node），海马体结构，以及小世界拓扑结构地图等来解决。我想可能是模仿人的记忆功能。

> Instead, a possible solution is to **define special interface nodes which sit between the real-time loop block and the map store**. These are nodes focused on **communication**, which are connected to the relevant components of real-time loop processing and then also to various sites in the map graph, and **may have some analogue in the hippocampus of mammal brains**. If the map store is **organised such that it has a ‘small world’ topology**, meaning that any part is joined to any other part by a small number of edge hops, then the interface nodes should be able to access (copy) any relevant map data in a small number of operations and serve them up to the real-time loop.
>
> **Each node in the map store will also have to play some part in this communication procedure, where it will sometimes beused as part of the route for copying map databackwards and forwards.**



### 注意力机制，主动视觉

这里的主动视觉是指机器人主动移动相机去采集和任务有关的信息，是一种“top-down”的执行方式。

> The active vision paradigm advocates using sensing resources, such as the directions that a camera points towards or the processing applied to its feed, **in a way which is controlled depending on the task at hand and prior knowledge available**. **This ‘top-down’ approach contrasts with ‘bottom-up’, blanket processing of all of the data received from a camera.**

Davison提到人的视觉机制是“bottom-up"和”top-down“并存的，而且现在的”bottom-up“的图像处理机制也有很不错的发展，而且在处理很多问题上都很有效，因此两者的结合应当也是一种必然，毕竟”top-down“的执行是需要信息和预测来作为先决条件的。主动视觉在系统的实时性上会有很大的帮助，因为减少了信息和数据处理的冗余度，只分析我需要的数据，因此会大大减小对算力的需求。

> **It is important that when assessing the relative efficiency of bottom-up versus top-down vision, we take into account not just processing operations but also data transfer, and to what extent memory locality and graph-based computation can be achieved by each alternative.** This will make certain possibilities in model-based prediction unattractive, such as searching large global maps or databases. The amazing advances in **CNN-based vision** means that we have to raise our sights when it comes to what we can expect from **pure bottom-up image processing**. But also, **graph processing will surely permit new ways to store and retrieve model data efficiently,** and will favour keeping and updating world models (such as graph SLAM maps) which have data locality.



## 总结

Davison这篇论文提出的思考和观点还是比较符合现在的主流认知的，而且在技术上，教授也给出了一些比较具体的方案。不过这个目标比较长远，目前其中的小环节可能都还没处理好，而且还需要硬件铺路，因此想要彻底实现难度还是有点大的。总之，这样的系统我估计未来都是模块化的，分布式的，并且是多协作的，以任务为中心的，毕竟现在的AI还没有大的突破，因此想要实现像人类那样的视觉机制还比较困难，得需要很多个学科的大佬共同研究努力才行。







