---
# type: docs 
title: Git 学习(四)
date: 2024-05-17T16:30:12+08:00
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

本章将给出一些复杂的样例，来熟练掌握git常用指令。

<!--more-->

## 复杂的实践操作

### 多次rebase

下图给出了初始树的状态和变化后的树的状态，要求只使用`git rebase`

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_4_1](Git/git_tutorial_4_1.png?fill=300x240,Left) | ![git_tutorial_4_2](Git/git_tutorial_4_2.png?fill=300x360,Left) |

步骤最少的命令如下:

1. `git rebase main bugFix`
2. `git rebase bugFix side`
3. `git rebase side another`
4. `git rebase another main`

下图展示了变化输入命令的变化情况：

![git_tutorial_4_3](Git/git_tutorial_4_3.gif)

### 两个parent节点

操作符`^`和`~`一样，后面都可以跟数字。`^`后面跟数字是指定合并提交记录的某个parent提交。这在某个位置上有多个parent提交的情况下十分有用。下面的例子展示了使用`git checkout main^`和`git checkout main^2`的情况:

|                            变化前                            |                     `git checkout main^`                     |                    `git checkout main^2`                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_4_4](Git/git_tutorial_4_4.png?fill=300x240,Left) | ![git_tutorial_4_5](Git/git_tutorial_4_5.png?fill=300x360,Center) | ![git_tutorial_4_6](Git/git_tutorial_4_6.png?fill=300x360,Center) |

此外，这种操作符也支持链式操作，比如`git checkout main~^2~2`。
