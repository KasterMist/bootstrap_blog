---
# type: docs 
title: Git 学习 (三)
date: 2024-05-15T13:48:56+08:00
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

本章将着重介绍如何更清晰的整理提交记录，git tag和git describe的用法，以及一些复杂的样例来熟练掌握git常用指令。

<!--more-->

## 整理提交记录

### git cherry-pick

`git cherry-pick <提交号>...`可以将一些提交复制到当前所在位置(HEAD)下面。

举个例子，假如我们想要将side分支上的工作复制到main分支，除了使用git rebase，使用git cherry-pick也是一种方法。下图是输入`git cherry-pick C2 C4`的变化:

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_3_1](Git/git_tutorial_3_1.png?fill=300x360,Left) | ![git_tutorial_3_2](Git/git_tutorial_3_2.png?fill=300x360,Left) |

可以看出，在main分支下输入`git cherry-pick C2 C4`后，main分支就会获取到C2和C4的提交记录。

但是，使用这种方式需要知道提交记录的哈希值，如果不清楚提交记录的哈希值的话，可以使用交互式的rebase。

### git rebase -i

-i是--interactive的简写，意思是交互式。在git rebase后添加了这个选项，git会打开一个UI界面并列出将要被复制到目标分支的备选提交记录，它还会显示每个提交记录的哈希值和提交说明，提交说明有助于你理解这个提交进行了哪些更改。实际使用时，显示的UI窗口一般会在文本编辑器(如vim)中打开一个文件。

在UI界面中我们可以:

- 调整提交记录的顺序
- 删除不想要的提交(通过切换pick的状态来完成，关闭就意味着你不想要这个提交记录)
- 合并提交(把多个提交记录合并成一个)

## git tag

分支很容易被人为移动，并且当有新的提交时，它也会移动。分支很容易被改变，大部分分支还只是临时的，并且还一直在变。而使用tag可以永远指向某个提交记录的标识。比如使用`git tag <tag name>`即可在当前分支创建一个tag。使用`git tag <tag name> <ref>`可以在对应引用的记录下创建一个tag。

## git describe

git describe可以用来描述最近的tag。

其语法为`git describe <ref>`。ref可以是任何能被git识别成提交记录的引用，如果你没有指定的话，git会使用你目前所在的位置(HEAD)。

输出结果为: `<tag>_<numCommits>_g<hash>`。tag表示的是离ref最近的标签，numCommits是表示这个ref与tag相差有多少个提交记录，hash表示的是你所给定的ref所表示的提交记录哈希值的前几位。当ref提交记录上有某个tag时，则只输出tag名称。

## 复杂的实践操作

### 多次rebase

下图给出了初始树的状态和变化后的树的状态，要求只使用`git rebase`

|                            变化前                            |                            变化后                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_3_3](Git/git_tutorial_3_3.png?fill=300x240,Left) | ![git_tutorial_3_4](Git/git_tutorial_3_4.png?fill=300x360,Left) |

步骤最少的命令如下:

1. `git rebase main bugFix`
2. `git rebase bugFix side`
3. `git rebase side another`
4. `git rebase another main`

下图展示了变化输入命令的变化情况：

![git_tutorial_3_5](/Users/xieletian/学习资料/博客信息/bootstrap_blog/content/posts/Git/Git/git_tutorial_4_5.gif)

### 两个parent节点

操作符`^`和`~`一样，后面都可以跟数字。`^`后面跟数字是指定合并提交记录的某个parent提交。这在某个位置上有多个parent提交的情况下十分有用。下面的例子展示了使用`git checkout main^`和`git checkout main^2`的情况:

|                            变化前                            |                     `git checkout main^`                     |                    `git checkout main^2`                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![git_tutorial_3_6](Git/git_tutorial_3_6.png?fill=300x240,Left) | ![git_tutorial_3_7](Git/git_tutorial_3_7.png?fill=300x360,Center) | ![git_tutorial_3_8](Git/git_tutorial_3_8.png?fill=300x360,Center) |

此外，这种操作符也支持链式操作，比如`git checkout main~^2~2`。
