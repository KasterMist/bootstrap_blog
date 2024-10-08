---
# type: docs 
title: Vim用法 (五) 深入理解Vim的寄存器
date: 2024-07-07T14:59:51+08:00
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
---

这一部分将探讨vim的寄存器的使用。

<!--more-->

## 复制与粘贴

vim的复制和删除都会将信息存入到寄存器中，所以说vim的删除可以视为剪切。

使用`xp`可以调换光标之后的两个字符。

使用`ddp`可以调换当前行与它的下一行。

使用`yyp`可以对当前行进行复制粘贴操作。

使用`yiw`可以对当前单词进行复制。

使用`diw`可以对当前单词进行删除(剪切)。



### 深入理解vim寄存器

加深对vim寄存器的理解以及使用，可以让我们更好的在vim中进行复制和删除。我们可以通过`"{register}`即`"`加上寄存器前缀的方式制定要用的寄存器。如果不指明，vim将缺省使用无名寄存器。

#### 有名寄存器("a-"z)

比如，我们可以使用`"ayiw`来将当前单词复制到寄存器`a`中，使用`"bdd`来将当前正行文本剪切到寄存器`b`中。之后，我们可以事情`"ap`、`"bp`来粘贴来自寄存器a和b的信息。

上述是普通模式的命令，我们也可以使用Ex命令来实现: 

- `:delete c`:把当前行剪切到寄存器c
- `:put c`: 将寄存器c粘贴至当前光标所在行之下

#### 无名寄存器

如果没有指定要用的寄存器，vim将缺省使用无名寄存器，可用`"`来表示，即`""p`等同于`p`的指令。换句话说，之前介绍的大部分删除剪切快捷键如果没有指明寄存器的话，都是将信息写入到无名寄存器中。所以，正常使用过程中无名寄存器非常容易被覆盖。

#### 复制专用寄存器("0)

复制专用寄存器当且仅当使用`y{motion}`的时候才会被赋值，使用`x`,`s`,`c{motion}`均不会覆盖该寄存器，所以复制寄存器是很稳定的，复制专用寄存器的前缀是`0`，即可以使用`"0p`来将复制专用寄存器中的信息粘贴。比如我们使用`yy`复制了一条信息，然后又使用了`diw`删除了一个单词，此时无名寄存器会存储`diw`表示的单词，而如果我们使用`"0p`则仍然会粘贴之前`yy`复制的信息。

#### 系统剪贴板("+)

如果想从vim复制文本到外部程序，必须要使用系统剪贴板。

在vim的复制或者删除命令之前加入`"+`，即可将相应的文本捕获至系统剪贴板。
