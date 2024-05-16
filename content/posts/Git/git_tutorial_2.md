---
# type: docs 
title: Git 学习 (二)
date: 2024-05-15T16:43:56+08:00
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
images: ["/images/logo/git.jpg"]
---

本章节将介绍git的一些高级操作，这些操作主要用来如何在提交树上进行移动，方便更灵活的更改分支以及提交节点。

<!--more-->

## 高级操作

### HEAD

我们有必要先学习在你项目的提交树上前后移动的几种方法。

HEAD是一个对当前所在分支的符号引用，也就是指向正在其基础上进行工作的提交记录。HEAD总是指向当前分支最近的一次提交记录。

HEAD通常情况下是指向分支名的(比如bugFix)。在提交时，改变了bugFix的状态，这一变化通过HEAD变得可见。

下面一个例子展现了使用`git checkout C1; git checkout main; git commit; git checkout C2`的变化:

![git_tutorial_2_1](Git/git_tutorial_2_1.gif#center)

分离HEAD就是让其指向了某个具体的提交记录而不是分支名。在命令执行之前的状态为: HEAD -> main -> C1，使用`git checkout C1`后，状态变为: HEAD -> C1，变化如下图所示:

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_2_2](Git/git_tutorial_2_2.png?fill=300x360,Left) | ![git_tutorial_2_3](Git/git_tutorial_2_3.png?fill=300x360,Right) |

### 相对引用

我们可以使用`git log`来访问提交记录的哈希值。通过指定提交记录哈希值来移动提交记录是可行的，但是由于哈希值在git中非常长，使用起来非常不方便。不过，git对哈希的处理很智能。你只需要提供能够唯一标识提交记录的前几个字符即可。因此我可以仅输入一部分字符来匹配。

此外，git引入了相对引用，便于移动提交记录。使用相对引用的话，可以在容易记忆的地方开始计算。

把操作符`^`加在引用名称的后面，表示让git寻找指定提交记录的parent提交。“main^”相当于“main的parent节点”，“main^^”相当于main的第二个parent节点。

下面展示的是使用`git checkout main^`发生的变化:

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_2_4](Git/git_tutorial_2_4.png?fill=300x360,Left) | ![git_tutorial_2_5](Git/git_tutorial_2_5.png?fill=300x360,Right) |

我们也可以将HEAD作为相对引用参照，下面的动图展示了`git checkout C3; git checkout HEAD^; git checkout HEAD^; git checkout HEAD^;`发生的变化:

![git_tutorial_2_6](Git/git_tutorial_2_6.gif#center)

使用`~<num>`向上移动多个提交记录，如`git checkout bugFix~3`就会在bugFix分支所在的记录一次性后退3步。

相对引用也可以用在强制修改分支位置的情况: 我们可以直接使用`-f`选项让分支指向另一个提交，比如`git branch -f main HEAD~3`，这种方式可以将main分支强制指向HEAD的第3级的parent提交。下图展示了变化情况:

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_2_6](Git/git_tutorial_2_6.png?fill=300x360,Left) | ![git_tutorial_2_7](Git/git_tutorial_2_7.png?fill=300x360,Right) |

### 撤销变更

有两种方法来撤销变更: `git reset`和`git revert`

#### git reset

`git reset`通过把分支记录回退几个提交记录来实现撤销改动。你可以将这想象成“改写历史”.`git reset`向上移动分支，原来指向的提交记录就跟从来没有提交过一样。

下面是输入`git reset HEAD～1`的变化情况:

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_2_8](Git/git_tutorial_2_8.png?fill=300x360,Left) | ![git_tutorial_2_9](Git/git_tutorial_2_9.png?fill=300x360,Right) |

git把main分支移回到C1，我们的本地代码库根本就不知道有C2这个提交了(在reset后，C2所做的变更还在，但是处于未加入暂存区状态)。

在本地使用git reset很方便，但是git reset无法对远程分支生效。

#### git revert

使用git revert可以撤销更改并分享给别人，下图展示了输入`git revert HEAD`后的变化，图中可以看出来我们要撤销的提交记录后面多了一个新提交，这是因为新提交记录C2引入了更改，这些更改刚好是用来撤销C2这个提交的。也就是说C2的状态与C1是相同的。revert之后就可以把更改推送到远程仓库和别人分享了。

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_2_10](Git/git_tutorial_2_10.png?fill=300x360,Left) | ![git_tutorial_2_11](Git/git_tutorial_2_11.png?fill=300x360,Left) |

