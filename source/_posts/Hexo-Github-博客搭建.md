---
title: Hexo+Github 博客搭建
mathjax: true
tags:
  - hexo
  - leancloud
  - next
categories: 博客搭建
abbrlink: 3102690793
date: 2019-02-14 17:26:38
---



**完整搭建过程参考[此系列博客](https://eirunye.github.io/categories/Hexo/)（环境win10）**

我的写作编辑器用的是[typora](<https://www.typora.io/>)

## Next主题配置

Next主题是本人比较中意的hexo主题，所以在此只记录Next主题的配置过程



## 下载

建议安装6.0版本。利用git bash here,输入命令：git clone https://github.com/theme-next/hexo-theme-next themes/next



## 配置与个性化



<!--more-->

1.添加作者头像并设置旋转效果

**注意：最新版本的next主题已经添加了头像动画功能，直接在主题的配置文件里面修改，因此头像添加（路径）也是在主题配置文件里面添加，而不是在根目录的配置文件添加**

```
# Sidebar Avatar
avatar: 
  # in theme directory(source/images): /images/avatar.gif
  # in site  directory(source/uploads): /uploads/avatar.gif
  # You can also use other linking images.
  url: /uploads/header.jpg     #/images/avatar.gif
  # If true, the avatar would be dispalyed in circle.
  rounded: true
  # The value of opacity should be choose from 0 to 1 to set the opacity of the avatar.
  opacity: 1
  # If true, the avatar would be rotated with the cursor.
  rotated: true
```
2.修改标签“#”符号和[插入图片](<https://blog.csdn.net/qq_37497322/article/details/80628713>)



3.添加社交账号 

如果遇到没有的社交图标，可以在[fontawesome](https://fontawesome.com/icons?from=io)中寻找添加。 ||后面的是图标名，建议不要找最新的。



4.菜单设置

本地搜索一旦开启会自动添加”搜索“菜单



5.添加版权信息。参考该[文章](http://stevenshi.me/2017/05/26/hexo-add-copyright/)

上面的链接是手动添加，现在可以在NEXT主题配置文件中进行修改

```
creative_commons:
  license: by-nc-sa
  sidebar: true
  post: true
  language:
```





6.利用leancloud加入阅读计数功能

若页面LeanCloud访问统计提示’Counter not initialized! See more at console err msg.’

参考该[博客](https://leaferx.online/2018/02/11/lc-security/)解决

**问题**：安装hexo-leancloud-counter-security报错：npm WARN babel-eslint@10.0.1 requires a peer of eslint@>= 4.12.1 but none is installed. You must install peer dependencies yourself.

因此缺什么就安装什么，在这里我是安装：npm install eslint@>= 4.12.1,之后再安装hexo-leancloud-counter-security就成功了。

另外最后在修正deploy的时候注意tab的控制，不然编译会报错

```
deploy:
  #type: git
  #repository: git@github.com:x695/thinkee.github.io.git
  #repository: https://github.com/x695/thinkee.github.io.git  
  #branch: master
  - type: git
    #repository: git@e.coding.net:yangshixian/blog.git
    repository: git@git.coding.net:thinkee/blog.coding.me.git
    branch: master
  - type: leancloud_counter_security_sync
```
然而，十分尴尬的是，阅读数依然没有显示，后来参考这篇[博客](https://leflacon.github.io/52b56662/)得到了解决



7.利用valine+leancloud添加评论功能

[参考]( https://bigwin.ml/2018/11/29/valine-for-next/ )



8.网站背景动画

如果修改主题配置文件之后不出现动画，建议下载主要的库到/source/lib中



9.加入本地搜索功能



10.加入Latex数学公式

这个除了在主题配置文件修改mathjax: true之外，**还需要在每个post的博客里的头代码里面加入mathjax: true才可以正常显示**。如果嫌每次书写都得添加麻烦，可以直接在根目录的scaffolds文件夹里的post.m文件夹直接加入mathjax: true，之后每次新建都会自动添加了。

不过再后续使用Latex语句的时候，发现博客上还是会显示源码，后来发现应该是渲染引擎的问题，根据[这篇博客](https://blog.csdn.net/qq_34229391/article/details/82725229)的指导修改后就好了。



------

------

2019.3.5日来更新，写完一篇blog进行generate时候出现了错误：

```
Template render error: (unknown path) [Line 62, Column 32]
  expected variable end
    at Object.exports.prettifyError 
```

出现的原因要么是主题配置文件忘了打空格，要么就是自己写的blog文件里面用的符号与其他hexo配置文件语法有冲突，后来发现原因是我打公式时出现了两个“}}”，前面一个"}"是latex语法，后面是"\}"，目的是为了输出括号“}”，之后参照latex的另一种打法“\lbrace \rbrace"就好了

------

------

---

2019.4.17日来更

公式直接用[Mathpix](<https://mathpix.com/>)进行LaTex命令复制吧，非常方便，也几乎没出现什么错误，还省时间。

---

11.博客图片点击放大

在NEXT主题的配置文件里面有fancybox选项，直接改为true就好了，不过在这之前要下载fancybox包到`next/source/lib`文件夹里面，即：

```
cd next/source/lib
git clone https://github.com/theme-next/theme-next-fancybox3 fancybox
```



12.并排插入两张图片

```
<figure class="half">
    <img src="xxx.jpg" width="400"/>
    <img src="xxx.jpg" width="400"/>
</figure>
```

不过这只能在.md文件中实现，博客中还是上下各一张。



13.NEXT主题[内置标签](<http://theme-next.iissnan.com/tag-plugins.html>)（包括引用文本居中，图片最大化引用等），更详细的标签见hexo[官网](<https://hexo.io/zh-cn/docs/tag-plugins.html>)。注意在本地利用这些标签不会出现相应的结果，只有上传到博客上才能看到。



14.添加emoji表情功能

[emoji大全](<https://www.webfx.com/tools/emoji-cheat-sheet/>)

关于加入表情的功能，我一共找到了两种方法，一种是基于修改渲染引擎的，然后加入twemoji插件，但是我在之前为了支持mathjax已经更换了渲染引擎，结果导致我在按照[博客1](<https://chaxiaoniu.oschina.io/2017/07/10/HexoAddEmoji/>)和[博客2](<https://www.cnblogs.com/fsong/p/5929773.html>)的操作进行时出现了fancybox图片放大和图片内部插入的问题，有时候甚至会影响到html语句，后来倒退删除渲染引擎和插件的时候，想要恢复到原来的状态，结果hexo g可以成功，本地却无法显示不出内容，结果只好重装。。目前原因未知。

此外，第二种方法是安装`hexo-filter-github-emojis`插件，可参考[博客3](<https://novnan.github.io/Hexo/emojis-for-hexo-next/>)和[博客4](https://www.biueo.com/2018/01/12/Hexo文章添加emoji表情/)，但是我看博客中也提到了图片干扰的问题，就没再继续试下去了。。。

下次看到更好的解决办法再来更新。



15.加入RSS订阅

直接在NEXT主题配置文件里面更改，在这之前先在博客站点目录下安装`npm install hexo-generator-feed --save`

```
feed:
    type: rss2
    path: rss2.xml
    limit: 5
    hub:
    content: 'true'
    
```



16.博客被谷歌和百度收录

参考这个[博客](<https://www.cnblogs.com/php-linux/p/8493346.html>)的做法

17.购买自己的阿里云域名并设置

[参考]( https://zhuanlan.zhihu.com/p/54575457 )

出现问题：`leancloud阅读统计和评论系统出现问题`，在`安全中心`更换了域名之后，deploy时出现Error，too much request，得知时免费版的限制导致错误，按照[此博客]( https://yunhao.space/2018/06/27/hexo-leancloud-plugin-installation-tutor/ )修改并重新配置了一个leancloud Counter应用未能解决问题。[此博客](https://www.bingyublog.com/2019/02/23/Hexo添加文章阅读量统计功能/) 通过加入lean-analytics.swig文件也未能解决问题。

**偶然通过换回valine+leancloud评论系统出现`code 403`问题进行改正时也顺便解决了这个问题。**

个人GitHub博客的`settings`页面的`Github Pages`有个` Enforce HTTPS  `选项，我没勾，所以http://或者https://都可以访问我的域名，而我的leancloud`安全中心`中的安全域名只加了https://的，在加一个http://就好了。。

但是deploy时依然会有些莫名其妙的错误，比如`ERROR Password must be string`，但是既然正常显示了，我就没再管了。。

18.更换评论Gitalk系统

由于leancloud经常出现问题，不易维护，所以可以考虑更换评论系统为Gitalk，虽然评论必须登陆GitHub账号，但是可以接受，毕竟现在理工科都有Github账号，此外还加了字数统计和阅读时长等信息。具体操作参见[链接]( https://yunhao.space/2018/06/29/hexo-next-function-setting/ )。

**需要注意，使用Gitalk评论系统时，每次deploy都需要在博客下登陆自己的账号初始化下**