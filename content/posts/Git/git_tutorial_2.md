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

本章节将介绍git的一些高级操作

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

此外，git引入了相对引用，便于移动提交记录。使用相对引用的话，可以在容易记忆的地方开始计算:

- 使用`^`向上移动一个提交记录
- 使用`~<num>`向上移动多个提交记录，如`~3`

把操作符`^`加在引用名称的后面，表示让git寻找指定提交记录的parent提交。“main^”相当于“main的parent节点”，“main^^“相当于main的第二个parent节点。

