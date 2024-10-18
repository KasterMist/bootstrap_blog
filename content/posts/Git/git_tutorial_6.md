---
# type: docs 
title: Git 学习 (六)
date: 2024-10-18T17:15:45+08:00
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

这部分将记录一些我遇到的实际的一些git流程，通过实际的问题可以快速了解哪些git命令是常用且重要的。由于每个人遇到的具体项目都不太一样，所以这里的一些情况只作为参考。

<!--more-->

### 解决版本冲突问题

假如我克隆了branch A的最新版本然后做出了一些代码的修改(并没有提交)，但是另一个人在branch A的基础上做了更新并提交到了remote上，现在branch A是最新版本，此时我想要获取最新的branch A，但是也不想放弃之前在本地更改的信息。

此时可以分为以下几步：

1. 保存本地的未提交修改

使用`git stash`可以将项目的修改保存到一个栈中，以便后续重新应用这些更改。

2. 更新远程分支

使用`git fetch origin`来更新远程分支

3. 合并远程分支的最新更改

使用`git chekout A`切换到分支A并使用`git merge origin/A`来合并到本地分支A中。

4. 重新应用本地修改

使用`git stash pop`可以获取到之前存在栈中的修改信息并应用到本地项目中。

5. 处理冲突

在使用`git stash pop`时很有可能有本地的修改与最新版本的冲突。此时可以使用`git status`来查看哪些文件有冲突，而且UI也会有冲突标记，之后就可以手动解决冲突。

6. 标记冲突文件为已解决

当解决完冲突文件后，可以使用`git add <file>`来告诉git这些文件的冲突已经解决。

7. 可选择进行合并

解决之后，可以选择`git commit`来合并项目并通过`git push`进行提交。
