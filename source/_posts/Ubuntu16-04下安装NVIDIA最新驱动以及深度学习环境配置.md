---
title: Ubuntu16.04下安装NVIDIA最新驱动以及深度学习环境配置
mathjax: true
tags:
  - deep learning
  - linux
  - nvidia driver
categories: 深度学习环境配置
abbrlink: 3059645377
date: 2019-02-18 18:15:39
---



---

----

---



<center>2019.3.8日来更，可以不用看以前下面写的内容了，因为不是很完整。。。</center>

如果按照之前说的通过官网下载的run文件来手动安装驱动，可能出现nvidia-settings打不开的情况，比如我就是输出信息：“ERROR: Unable to load info from any available system”，这样会导致我的电脑识别不了外接显示器。。。

后来在网上看了一些人的回答，大致的原因是官网的run文件在安装时后会默认更改一些系统配置，因此可能会导致一些错误。。

**解决办法：通过Ubuntu自动的install功能进行安装**

1.先按照这个[教程](https://blog.csdn.net/u014561933/article/details/79958017)卸载驱动，然后禁用好nouveau；

2.重启后确认nouveau已禁用，关闭图形界面，进入命令行界面；

3.输入下面的命令：

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia- #这里选择你要安装的版本，最新的应该也没问题，我就是直接装的410，省得到时候又得重装CUDA和CUDNN
reboot #重启
```

4.再次开机后应该就没问题了，nvidia-smi, nvidia-settings都没问题了，外接显示器也正常了。

---

---

---



## 写在前面

本人电脑是战神神舟Z7M-KP5GZ，显卡是GTX1050TI，按照以前利用liveusb的方法安装了Ubuntu16.04双系统，然而第一次安装NVIDIA驱动的时候有些问题，直接是在软件更新里面安装的系统自带驱动，版本384。后来配置深度学习环境需要安装CUDA和CUDNN，由于安装最新版本的CUDA需要相应的驱动版本对应，因此需要对NVIDIA驱动进行更新。

在安装过程中，遇到了一些问题，主要是：

- 驱动安装后Ubuntu系统无法进入，卡在启动界面的5个原点；

- 成功进入系统后，分辨率低；

后来根据几篇博客解决了上述问题：

- [博客1](https://blog.csdn.net/wang_ys121/article/details/82881716)

- [博客2](https://blog.csdn.net/xunan003/article/details/81665835)

- [博客3](https://blog.csdn.net/ezhchai/article/details/78788564)

- [博客4](https://blog.csdn.net/yinxingtianxia/article/details/82503388)

后来总结了下，**最主要的问题是一定要禁用nouveau**，可以通过下面的命令看是否禁用，如果不输出信息则是成功禁用。。一定要注意，如果之前装过了NVIDIA驱动，然后卸载重装了一定要记得再次禁用！！！！

```
lsmod | grep nouveau
```

<!--more-->

## 利用run方式安装NVIDIA最新驱动

**1.验证nouveau是否已禁用**

```
lsmod | grep nouveau
```





2.如果没有输出信息，则说明已经禁用，如果没有，编辑文件blacklist.conf

```
sudo vim /etc/modprobe.d/blacklist.conf
```

在文件最后部分插入以下两行内容(i插入，esc退出编辑，:wq退出并保存)

```
blacklist nouveau
options nouveau modeset=0
```

更新系统

```
sudo update-initramfs -u
```

重启系统，再看看是不是禁用了nouveau







3.在[英伟达官网](<http://www.nvidia.cn/page/home.html>)查找并下载自己对应的驱动(.run文件)







4.ctrl+alt+F1进入命令行界面，登陆进入，并关闭图形界面

```
sudo service lightdm stop
```





5.卸载原有的驱动

```
apt-get purge nvidia-*
nvidia-smi
#如果还有信息，输入下面的代码
sudo /usr/bin/nvidia-uninstall
nvidia-smi
#没有驱动信息则成功
```



6.进入驱动文件夹

```
sudo chmod a+x NVIDIA-Linux-x86_64-xxx.x.run #这里版本号每个人可能不同(xxx.xx代表版本号)
sudo ./NVIDIA-Linux-x86_64-xxx.xx.run -no-x-check -no-nouveau-check -no-opengl-files
```





**7.安装过程中的选项很重要，如果选错了，不要重启，重新安装就行了**

**其中register the kernel module souces with DKMS选择NO，Nvidia's 32-bit compatibility libraries也选NO，其他的YES或者正常就行**。





8.检查是否安装完成

```
modprobe nvidia #挂载驱动
nvidia-smi
#如果出现驱动信息表示安装成功
sudo service lightdm start #重新进入图形界面
```





**9.如果重启开机没问题，那么万事大吉，如果进不去，或者卡在启动界面，可以选择recovery mode进入系统，然后修改grub文件，我是依据[博客3](https://blog.csdn.net/ezhchai/article/details/78788564)解决了我的问题，我看了下，应该还是nouveau的问题。**

**每个电脑不一样，如果依据此博客还是解决不了，那么根据自己的问题去百度或者Google吧。。**







10.通过recovery mode进入了系统后，重新按照前述步骤卸载重装NVIDIA驱动，这说明了我之前卸载不干净或者卸载之后nouveau又被重新开启了。。。。







11.之后重启后就能正常进入Ubuntu系统了，这说明应该已经成功了，之后再**重新打开grub文件（[博客3](https://blog.csdn.net/ezhchai/article/details/78788564)），把之前改过的再改回来**，然后重启









12.重启之后发现分辨率低而且无法在设置中心改，按照[博客4](https://blog.csdn.net/yinxingtianxia/article/details/82503388)的方法结果成功解决了问题，自此，NVIDIA驱动安装完成。









## 安装CUDA和CUDNN

安装好驱动后基本上等于深度学习环境配置进度完成了80%了，之后可以安装相应的CUDA和CUDNN了，安装过程比较简单，在此不再赘述，可以参照以下两篇博客，也可以自己重新定义版本关键词进行搜索。

[博客6](https://blog.csdn.net/wanzhen4330/article/details/81699769)

[博客7](https://blog.csdn.net/weixin_42279044/article/details/83181686)





------

------

<center> **2018.3.2踩坑来更** </center>

------

------



今天Ubuntu系统提示升级，结果升级了内核kernel，导致升级完成重启后出现ACPI ERROR之类的错误，看了一大推的博客总结后，发现问题是**“升级之后的内核，是不会自动加载你的显卡驱动的，那就需要在这个心内核上手动重新安装NV驱动。”** 所以，等装完了驱动后，最好禁止Ubuntu内核更新或者直接禁止Ubuntu更新。

所以，升级内核后，需要重新安装NVIDIA驱动。。！！！！！！

**当然重装驱动的最重要步骤就是禁用nouveau,这里我看到了另一篇博客（[博客8](https://blog.csdn.net/dihuanlai9093/article/details/79253963/)）讲得比较详细，基本上只看这一篇博客就可以安装并配置好深度学习环境了，要记得同时修改/etc/modprobe.d/disable-nouveau.conf和/etc/default/grub**

而且要特别注意，装完后要记得回来修改/etc/default/grub，否则修改分辨率后会发现没有作用。我今天安装时做了实验，我是按照[博客3](https://blog.csdn.net/ezhchai/article/details/78788564)对/etc/default/grub进行修改的，也就是加了“nomodeset”，发现只有将它删除，[博客4](https://blog.csdn.net/yinxingtianxia/article/details/82503388)的修改方法才有用，另外注意[博客4](https://blog.csdn.net/yinxingtianxia/article/details/82503388)添加的Modes后面的分辨率要根据自己的电脑调整，否则可能出现登录界面比较小，没接到电脑边。

英伟达的驱动真是一大堆破事，，看来是时候着手docker了。。











