---
# type: docs 
title: Git 学习 (五)
date: 2024-05-22T13:53:59+08:00
featured: false
draft: false
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Tutorial"]
tags: ["Git"]
images: ["images/logo/git.jpg"]
---

本章将介绍一下远程追踪功能以及git push, git fetch, git pull等命令的参数设置。

<!--more-->

## 远程追踪

通过之前的演示可以发现，main和o/main(origin/main)似乎是关联的，实际上main和o/main的关联关系就是由分支的“remote tracking”属性决定的。main被设定为跟踪o/main —— 这意味着为main分支指定了推送的目的地以及拉取后合并的目标。

当克隆仓库的时候，这个属性已经设置好了。当进行克隆时, git会为远程仓库中的每个分支在本地仓库中创建一个远程分支（比如o/main）。然后再创建一个跟踪远程仓库中活动分支的本地分支，默认情况下这个本地分支会被命名为main。

我们也可以自己定一这个属性，有两种方法：

1. `git checkout -b totallyNotMain o/main`: 就可以创建一个名为totallyNotMain的分支，它跟踪远程分支o/main。
2. `git branch -u o/main totallyNotMain`: 这种方式也可以实现，不过前提是totallyNotMain分支需要存在。如果已经在这个分支上，也可以省略totallyNotMain名称，即`git branch -u o/main`。

## git命令参数设置

### git push参数

`git push <remote> <place>`: 比如`git push origin main`，意思是切到本地仓库中的main分支，获取所有的提交，再到远程仓库origin中找到main分支，将远程仓库中没有的提交记录都添加上去。

通过`<place>`参数来告诉git提交记录来自于main，要推送到远程仓库中的main。它实际上就是要同步的两个仓库的位置。通过指定参数告诉了git所有它需要的信息, 它会忽略目前所切换分支的属性。

下面一个例子可以比较直观的展现`git push <remote> <place>`的用途。使用`git checkout C0; git push origin main`会产生下面的变化。虽然我们将HEAD移到了C0处，但由于设置了git push的对应的分支信息，远程仓库的main分支会得到更新。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_5_1](Git/git_tutorial_5_1.png?fill=300x360,Left) | ![git_tutorial_5_2](Git/git_tutorial_5_2.png?fill=300x360,Left) |

如果我们使用`git checkout C0, git push`，即并不对git指定参数的话，并不会成功进行提交，因为我们切换的HEAD并没有跟踪任何的分支。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_5_1](Git/git_tutorial_5_1.png?fill=300x360,Left) | ![git_tutorial_5_3](Git/git_tutorial_5_3.png?fill=300x360,Left) |



如果想要在`git push`中同时为source和destination指定`<place>`的话，可以使用":"将二者连接起来。

`git push origin <source>:<destination>`: source可以是任何git识别的位置，比如`git push origin foo^:main`，git将foo^解析为一个位置，上传所有未被包含到远程仓库里main分支中的提交记录。如果要推送的destination的分支并不存在的话，git会在远程仓库中根据提供的名称创建这个分支。

### git fetch参数

git fetch的参数与git push非常类似，只不过是将顺序反转。比如`git fetch origin <source>:<destination>`中\<source>是远程仓库的分支位置，\<destination\>是本地仓库的分支位置。

### git push与git fetch空source的情况

我们可以在`git push`和`git fetch`的时候不指定任何的source，比如:

- `git push origin :side`: 这个命令会删除远程仓库的side分支(本地仓库的o/side也会被删除)
- `git fetch origin :bugFix`: 这个命令会fetch空到本地，也就是会在本地创建一个名字叫bugFix的分支

### git pull参数

因为`git pull`可以视为`git fetch`和`git merge`的缩写，所以可以理解为用同样的参数执行`git fetch`然后再用`git merge`合并抓取到的提交记录。

比如:

- `git pull origin foo`等效于`git fetch origin foo; git merge o/foo`
- `git pull origin bar:bugFix`等效于`git fetch origin bar:bugFix; git merge bugFix`

