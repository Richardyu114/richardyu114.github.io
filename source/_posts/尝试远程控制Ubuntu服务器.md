---
title: 尝试远程控制Ubuntu服务器
mathjax: true
tags:
  - Ubuntu
  - ssh
  - git
categories: 技术支持
abbrlink: 58897936
date: 2020-01-05 19:45:06
---

## About

暑假实习用公司的显卡的经历很舒服，所以寒假回家之前，想在教研室的工作站上搭个ssh，方便回家也可以用显卡跑网络。但是自己对这方面知识不是很了解，尤其是端口映射之类的操作，所以导致外网登陆服务器的时候折腾了一点时间，最后勉强利用ngork进行了内网穿透。。。使用过程中发现，ngork似乎有些延迟，加上scp传文件问题（我是win利用git bash登Ubuntu服务器）一直没解决，还是放弃了ssh转战了teamviewer（还是有点香的。。）

<!--more-->

## ssh

我安装的时候没关心密匙之类的问题，也没去修改ssh 的config文件，直接在Ubuntu 18.04上安装openssh-server就好了。

在Ubuntu终端上输入：

```
sudo apt-get update
sudo apt-get openssh-server
# 启动ssh
sudo service ssh start
# 重启
sudo service ssh restart
# 查看有没有启动
ps -aux | grep ''ssh''
# 关闭ssh
sudo /etc/init.d/ssh stop  
```

然后我在windows上打开git bash，输入：

![登陆命令](尝试远程控制Ubuntu服务器\log_command.PNG)

`-p`代表端口，默认是22，如果需要修改可以去`/etc/ssh/sshd_config`修改（改过记得source一下生效），参考[blog1](https://blog.mythsman.com/post/5d2d65fca2005d74040ef873/)和[blog2](https://www.jianshu.com/p/7028e5fecf2b)。端口后面跟的是`uername@ip_of_server`，密码输入服务器用户名密码，如果和服务器处于一个局域网，你的windows是可以无缝登陆的。服务器的ip地址可以通过`ifconfig`查询。

但是只能在一个局域网登陆没什么用处。。我们需要随时随地登陆服务器，这时候就需要科普一些公网，内网，端口映射等知识，看得我也有点眼花...后来我就找了个比较简单的内网穿透方法，也没有自己买服务器降低延迟啥的，基本上按照[blog3](https://zhuanlan.zhihu.com/p/60962957)的教程，去[ngork](https://ngrok.com/)下载软件，然后解压，在输入命令`./ngork tcp 22`，这个`tcp 22`默认端口22。在`Forwarding`一行会出现地址（`0.tcp.ngrok.io` ）和端口号，同样地在git bash输入即可。

## Teamviewer

没自己买服务器做内网穿透的结果就是卡！卡！延迟让我有点受不了，而且scp一直搞不到win上来，也懒得查是不是自己命令哪里错了。当然除了上面的ngork外还可以搞个VPN，让你的电脑连上服务器所在的网，然后就可以按照最开始的方式登陆和传输了，暑假实习的公司就是这么干的，不过他们有专人维护这块，稳定性以及文件传输速度都很快。

最后.....

我还是转向了teamviewer，设置了固定密码。

![这延迟还能接受，传输速度也不赖，还要啥自行车...](尝试远程控制Ubuntu服务器\teamviewer.PNG)

