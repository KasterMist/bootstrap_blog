---
# type: docs 
title: Git 学习 (四)
date: 2024-05-20T17:04:57+08:00
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

本章将详细介绍如何使用git push和git pull，以及远程仓库的一些知识。

<!--more-->

## 远程仓库的用法

### git clone

从技术上来讲，`git clone`命令在真实的环境下的作用是在本地创建一个远程仓库的拷贝。常用用法有: `git clone [url]` ,`git clone [url] --branch [branch]`等。

克隆远程分支时，我们的本地仓库里面会出现一个名为"origin/main"的分支(主要的远程仓库默认命名为"origin")，这种类型的分支叫作远程分支。切换到远程分支时，自动进入分离HEAD状态(也就是"origin/main"并不会进行更新)，因为git不能直接在这些远程分支上进行更改，我们必须在别的地方完成你的工作,(更新了远程分支之后)再用远程分享成果。

### git fetch

git远程仓库相当的操作实际可以归纳为两点: 

1. 向远程仓库传输数据
2. 从远程仓库获取数据

`git fetch`用来从远程仓库获取数据。当我们从远程仓库获取数据时, 远程分支也会更新以反映最新的远程仓库。

下面是一个使用`git fetch`的实例。图中有一个远程仓库，它有两个本地仓库没有的提交。使用`git fetch`后，C2和C3被下载到了仓库，同时远程分支`o/main`也被更新，反映到了这一变化。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_4_1](Git/git_tutorial_4_1.png?fill=300x360,Left) | ![git_tutorial_4_2](Git/git_tutorial_4_2.png?fill=300x360,Right) |

`git fetch`完成了仅有的但是很重要的两步:

- 从远程仓库下载本地仓库中缺失的提交记录
- 更新远程分支指针(o/main)

`git fetch`通常通过互联网(使用http://或git://协议)与远程仓库通信。

而`git fetch`并不会改变你本地仓库的状态。它不会更新你的main分支，也不会修改你磁盘上的文件。我们可以将`git fetch`的理解为单纯的下载操作。

### git pull

当远程分支中有新的提交时，我们可以像合并本地分支那样来合并远程分支。实际上，由于先抓取更新再合并到本地分支这个流程很常用，我们可以使用`git pull`来完成这两个操作。`git pull`也可以理解为是`git fetch`和`git merge`的缩写。下图产生的变化可以通过`git pull`或`git fetch`和`git merge`生成。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_4_3](Git/git_tutorial_4_3.png?fill=300x360,Left) | ![git_tutorial_4_4](Git/git_tutorial_4_4.png?fill=300x360,Right) |

### git push

`git push`负责将本地的变更上传到指定的远程仓库，并在远程仓库上合并你的新提交记录。一旦`git push`完成, 其他人就可以从这个远程仓库下载新的提交成果(注:`git push`不带任何参数时的行为与git的一个名为`push.default`的配置有关。它的默认值取决于你正使用的git的版本)。

下图展示了使用`git push`后的变化。远程仓库接收了C2，远程仓库的main分支指向了C2，本地文件的远程分支(o/main)也同样进行了更新，即所有的分支都完成了同步。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_4_5](Git/git_tutorial_4_5.png?fill=300x360,Left) | ![git_tutorial_4_6](Git/git_tutorial_4_6.png?fill=300x360,Right) |

### 处理偏移的提交

假如我们克隆了一个仓库，此时的提交是C1，当我们研发了新的功能准备提交时(记为C3)，远程仓库的提交已经变成C2了，此时我们使用`git push`的时候就不能直接将C3直接提交到远程仓库中，因为远程仓库最新的提交是C2，而我们的提交C3是基于之前版本的C1实现的。我们需要先合并远程最新的代码，然后才能分享。比如下面的图，我们使用`git push`时，最新提交的C3基于远程分支中的C1，而远程仓库中该分支已经更新到C2了，所以git会拒绝推送请求。

![git_tutorial_4_7](Git/git_tutorial_4_7.png?width=300px#center)

#### 解决方法

##### 使用git rebase

最直接的方法是通过`git rebase`调整工作。

下图是采用了`git fetch; git rebase o/main; git push`的变化: 使用`git fetch`会更新本地仓库中的远程分支，然后用 rebase 将我们的工作移动到最新的提交记录下，最后再用`git push`推送到远程仓库。

![git_tutorial_4_8](Git/git_tutorial_4_8.gif?width=400px#center)

##### 使用git merge

使用`git merge`也可以进行解决。

下图是采用了`git fetch; git merge o/main; git push`的变化: 使用`git fetch`更新了本地仓库中的远程分支，然后合并了新变更到我们的本地分支(为了包含远程仓库的变更)，最后用`git push`把工作推送到远程仓库。

![git_tutorial_4_9](Git/git_tutorial_4_9.gif?width=400px#center)

##### 使用git pull

上面提到过`git pull`是`git fetch`和`git merge`的简写，`git pull --rebase`是`git fetch`和`git rebase`的简写。也就是说，使用`git pull`或者`git pull --rebase`就跟上面的结果是一样的。

### 远程服务器被拒绝

如果出现远程服务器被拒绝的情况，很可能是main被锁定了，需要一些Pull Request流程来合并修改。如果直接提交(commit)到本地main，然后试图推送(push)修改，你将会收到这样类似的信息: `![远程服务器拒绝] main -> main (TF402455: 不允许推送(push)这个分支; 你必须使用pull request来更新这个分支.)`

出现这种情况的原因可能是远程服务器拒绝直接推送(push)提交到main，因为策略配置要求pull requests来提交更新。

应该按照流程，新建一个分支，推送(push)这个分支并申请pull request。如果忘记并直接提交给了main，就会出现远程服务器被拒绝的情况。

解决方法就是新建一个分支推送到远程服务器，然后reset你的main分支和远程服务器保持一致, 否则下次使用`git pull`并且他人的提交和你冲突的时候就会有问题。
