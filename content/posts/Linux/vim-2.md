---
# type: docs 
title: Vim用法 (二) -- 模式便捷使用及拓展
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

大部分Vim命令都是在非插入模式中执行，不过插入模式中仍有一些能够提高效率的功能。此外，可视模式中也有一些快捷方式来快速选区区域。这篇博客将主要添加一些不那么常用的功能，根据需求来选择是否使用。

<!--more-->



## 插入模式中及时更正错误 (不常用)

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



## 粘贴寄存器中的文本 (有些时候很有用)

如果在visual模式下使用`"+y`把文本放入了寄存器中，可以在插入模式下使用`<Ctrl-r>0`来把寄存器中信息粘贴到光标位置。



## 插入模式中执行运算 (不常用)

我们可以在插入模式中，使用`<Ctrl-r>=` (Ctrl键加上r键再加上=键)来输入计算表达式，输入完后按下回车键即可在将结果插入到当前光标下。



## 可视模式快捷选取区域 (自定义修改区域时很有用)

我们可以在可视模式中对精确的文本对象进行选区。假如当前光标在某个区域内，比如`{}`中，我们可以使用快捷键来选区整个`{}`区域。下面是一些快捷键使用方法 (默认下面的操作是已经在可视模式下进行，即已经按键`v`之后)。

我们可以将快捷键中的`i`理解为inside，即覆盖某个区域内部的信息，将`a`理解为around，即也包括了区域标识符。标识符比如`(`与`)`的意思相同，可以进行替换。

| 按键 | 内容                                 |
| ---- | ------------------------------------ |
| `i}` | 选中{}内部的文本，不包括{}           |
| `a"` | 选中""内部的文本，包括""             |
| `i>` | 选中<>内部的文本，不包括<>           |
| `it` | 某个xml标签的内部文本，不包含xml标签 |
| `at` | 某个xml标签的内部文本，包含xml标签   |



