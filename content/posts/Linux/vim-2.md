---
# type: docs 
title: Vim用法 (二)
date: 2024-06-12T16:16:02+08:00
featured: false
draft: false
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories:     ["Note"]
tags:    		["Linux", "Linux Command"]
images:    		["/images/logo/vim.png"]
authors:
---

大部分Vim命令都是在非插入模式中执行，不过插入模式中仍有一些能够提高效率的功能。这篇博客将主要介绍插入模式的一些有用的功能。

<!--more-->

## 插入模式中及时更正错误

除了使用退格键，我们还可以使用Ctrl+字符来实现删除特定的信息:

- `<Ctrl-h>`: 删除前一个字符(同退格键)
- `<Ctrl-w>`: 删除前一个单词
- `<Ctrl-u>`: 删至行首

这些命令在bash shell中也可以使用



## 普通模式

- `Esc`: 切换到普通模式
- `<Ctrl-[>`: 切换到普通模式
- `<Ctrl-o>`: 切换到插入-普通模式

插入-普通模式是一种特殊的模式，它可以让我们执行一次普通模式命令，然后自动切换回插入模式



## 粘贴寄存器中的文本

如果在visual模式下使用`"+y`把文本放入了寄存器中，可以在插入模式下使用`<Ctrl-r>0`来把寄存器中信息粘贴到光标位置。



## 运算

我们可以在插入模式中，使用`<Ctrl-r>=` (Ctrl键加上r键再加上=键)来输入计算表达式，输入完后按下回车键即可在将结果插入到当前光标下。
