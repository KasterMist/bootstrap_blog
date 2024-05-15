---
# type: docs 
title: Git 学习 (一)
date: 2024-05-14T17:02:07+08:00
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
images: []
---

本文章根据"Learn Git Branching"网站进行记录知识点。该网站将git流程可视化，可以很好的学习git相关知识。如果有兴趣，可以访问该网站: https://learngitbranching.js.org/?locale=zh_CN

<!--more-->

## 基础指令

### git commit

`git commit`作为常用的git命令之一，其功能可以视为一个提交记录。每次进行提交时，它并不会盲目地复制整个目录。条件允许的情况下，它会将当前版本与仓库中的上一个版本进行对比，并把所有的差异打包到一起作为一个提交记录。同时，Git还保存了提交的历史记录，我们使用git commit的信息都会有保留。

我们可以使用`git commit -m "message"`来对提交记录进行一个描述。

### git branch

`git branch`可以创建一个新的分支，我们可以在新的分支里面进行更新信息而不会影响原有的分支。当更新新的分支时，我们需要在新的分支下面才能更新。

- 创建新的分支: `git branch <name>`
- 使用`git checkout <name>`可以切换到新的分支上。
- 创建一个分支同时切换到该分支: `git checkout -b <name>`

### git merge

`git merge`用于将分支进行合并。我们可以创建一个新的分支，在其上面开发，开发完后再合并回主线。在 Git 中合并两个分支时会产生一个特殊的提交记录，它有两个 parent 节点。翻译成自然语言相当于：“我要把这两个 parent 节点本身及它们所有的祖先都包含进来。”

如下图所示，现在有两个分支，每个分支都有一个独自的提交。这样的话没有一个分支包含了所有的提交，我们可以使用`git merge`来进行合并。如果我们当前是在main分支下，我们可以使用`git merge bugFix`来把bugFix合并到main里面。

![git_tutorial_1_1](Git/git_tutorial_1_1.png?width=300px&height=360px)![git_tutorial_1_2](Git/git_tutorial_1_2.png?width=300px&height=360px#float-end)

现在从新的 main 开始沿着箭头向上看，在到达起点的路上会经过所有的提交记录。这意味着 main 包含了对代码库的所有修改。

如果再把 main 分支合并到 bugFix 分支，就会让两个分支都包含所有的修改，使用`git checkout bugFix`和`git merge main`即可实现，如下图所示:

![git_tutorial_1_3](Git/git_tutorial_1_2.png?width=300px&height=360px)![git_tutorial_1_2](Git/git_tutorial_1_3.png?width=300px&height=360px#float-end)

### git rebase

`git rebase`是另一种合并方法，Rebase 实际上就是取出一系列的提交记录，“复制”它们，然后在另外一个地方逐个的放下去。Rebase 的优势就是可以创造更线性的提交历史。

如下图所示，现在有两个分支，我们现在在 bugFix 分支上。如果我们想让 bugFix 分支里的工作直接移到 main 分支上，移动以后会使得两个分支的功能看起来像是按顺序开发，但实际上它们是并行开发的。我们可以使用`git rebase`来进行实现。此时 bugFix 分支上的工作在 main 的最顶端，同时我们也得到了一个更线性的提交序列。

值得注意的是，提交记录C3仍然存在，C3'是我们rebase到 main 分支的C3副本。

![git_tutorial_1_4](Git/git_tutorial_1_4.png?width=300px&height=360px) ![git_tutorial_1_5](Git/git_tutorial_1_5.png?width=300px&height=360px#float-end)

剩下的操作就是更新 main 分支，将 main 分支和 bugFix 分支同步。使用`git checkout main`切换到 main 分支，然后`git rabase bugFix`即可，由于 bugFix 继承 main ，所以git只是简单的把 main 分支的引用向前移动了一下。变化如下图所示:

![git_tutorial_1_5](Git/git_tutorial_1_5.png?width=300px&height=360px) ![git_tutorial_1_6](Git/git_tutorial_1_6.png?width=300px&height=360px#float-end)

