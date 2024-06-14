---
# type: docs 
title: Vim用法 (三) -- 管理多个文件
date: 2024-06-13T16:57:11+08:00
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

这部分将着重介绍一下如何使用vim打开多个文件以及如何管理打开的文件。

<!--more-->

## 使用缓冲区打开多个文件

比如在包含多个.txt的路径下调用`vim *.txt`可以打开多个txt文件。此时窗口内的缓冲区对应着第一个文件，其他文件在窗口中是不可见的。

可以通过`:ls`来查看缓冲区的文件列表。

切换显示缓冲区的文件：

- `:bprev`: 反向遍历缓冲区的文件 (显示当前显示文件的上一个文件)
- `:bnext`: 正向遍历缓冲区的文件 (显示当前显示文件的下一个文件)
- `:bfirst`: 跳转到缓冲区文件列表的开头
- `:blast`: 跳转到缓冲区文件列表的结尾

删除缓冲区的某个文件: `bdelete N1`，N1是在`:ls`后显示的文件的序号，同时也可以直接输入文件名进行删除。



## 将工作区切分称窗口

vim启动时只会打开单个窗口。下面介绍几个水平切分和垂直切分窗口的方法：

- `<Ctrl-w>s`: 水平切分窗口，两个窗口显示相同的文件信息，同`:sp`
- `<Ctrl-w>v`: 垂直切分窗口，两个窗口显示相同的文件信息，同`:vsp`
- `:edit {filename}`: 在当前窗口下打开另一个文件，即切换为另一个文件显示。
- `:sp {filename}`: 水平切分窗口，新的窗口打开的文件是filename.
- `:vsp {filename}`: 垂直切分窗口，新的窗口打开的文件是filename.

窗口之间切换:

- `<Ctrl-w>w`: 在窗口间循环切换
- `<Ctrl-w>h`: 切换到左边的窗口
- `<Ctrl-w>j`: 切换到右边的窗口
- `<Ctrl-w>k`: 切换到上边的窗口
- `<Ctrl-w>l`: 切换到下边的窗口

实际上，`<Ctrl-w><Ctrl-w>`完成的功能和`<Ctrl-w>w`相同，如果想要多次切换活动窗口的话，一种简单的方法就是按住`Ctrl`键后输入`ww` (或者`wj`等其他切换命令)。

可以使用`:qa`,`:qa!`关闭所有的窗口。



## 用标签页将窗口分组

vim同样可以像其他IDE一样是用标签页来管理多个文件。vim的标签页更像是Linux中的虚拟桌面，新的标签页并不会打乱之前标签页设置的窗口排版。

打开和关闭标签页

- 打开标签页: `:tabe {filename}`，是`:tabedit {filename}`的简写。
- 关闭当前标签页: `:tabc`，是`:tabclose`的简写。
- 只保留当前标签页，关闭其他标签页: `:tabo`，是`:tabonly`的简写。

切换标签页

- 切换到编号为{N}的标签页: `:tabn {N}`，是`:tabnext {N}`的简写。普通模式下也可以使用`{N}gt`来跳转。
- 切换到下一标签页: `:tabn`，是`:tabnext`的简写。普通模式下也可以使用`gt`来跳转。
- 切换到上一个标签页: `:tabp`，是`tabprevious`的简写。普通模式下也可以使用`gT`来跳转。

重排标签页

- 使用`:tabmove {N}`即可将当前标签页排序到N的位置处。当{N}为0时，当前标签页会被移到开头。当省略了{N}，当前标签页会被移动到结尾。
