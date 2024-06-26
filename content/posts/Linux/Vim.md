---
title: 		 	"Vim用法 (一)"
subtitle:    	""
description: 	""
date:        	2024-04-13
#type:   		docs
featured:    	true
draft:    		false
comment:    	true
toc:    		true
reward:    		true
pinned:    		false
carousel:    	true
categories:     ["Note"]
tags:    		["Linux", "Linux Command"]
images:    		["/images/logo/vim.png"]
authors:
---

本章节将介绍vim的常用语法。

<!--more-->



在命令行中可以输入`vimtutor`来进行vim学习。

vim学习参考视频链接:
https://www.bilibili.com/video/BV1PL411M7bg/?spm_id_from=333.788&vd_source=302ea1a1118a80c10a5b35e58bd9c8bf

创建或编辑一个文件: `vim filename` (如果filename不存在，则创建该文件)

## 常用vim语句

### vim的三种模式

1. 普通模式 (Normal mode) / 命令模式 (Command mode): 默认模式，用于导航、删除、复制等操作。
2. 插入模式 (Insert mode): 用于输入文本。
3. 视图模式 (Visual Mode): 用于文本搜索、筛选。

在普通模式下按下`i`即可进入插入模式，按下`v`即可进入视图模式。在插入模式或者视图模式下按下`esc`即可退出该模式并进入普通模式。

**模式转换的快捷键使用**

普通模式-->插入模式

- 在下方插入一行: `o`
- 在上方插入一行: `O`(大写的o)
- 在当前光标后插入: `a`
- 在当前光标前插入: `i`
- 在行尾插入: `A`
- 在行首插入: `I`
- 特殊模式 replace mode: 输入`R`即可进入replace模式，此时输入的字符会替换当前字符(即先删除当前字符再输入新的字符)

### 光标移动

基础部分

- 上下左右: 左: `h` 下: `j`  上: `k`  右: `l`
- 移动到第一行: `gg`
- 移动到做后一行/指定行: `G`  行数+大写G跳到指定行

进阶部分

- 移动到下一个单词的开头: `w`
- 移动到下一个单词的结尾: `e`
- 移动到上一个单词的开头: `b`
- 移动到上一个单词的结尾: `ge`
- 注意，上面这四个使用方式有大写的方法，如果为大写，则表示的是字串 (可以理解为两个空格之间的信息)，小写表示的是单词。
- 移动行首: `0` (数字零)
- 移动到第一个非空字符: `^`
- 移动到行尾: `$`
- 移动到匹配的括号处: `%`
- 移动到变量定义处: `gd`
- 移动到前一个没有匹配的左大括号处: `[{`
- 移动到下一个没有匹配的右大括号处: `]}`
- 修改大小写(如果是大写，则修改为小写。如果是小写，则修改为大写): `~`

### 保存与退出

- 保存: `:w`
- 退出: `:q`
- 保存+退出: `:wq`
- 强制退出(不保存): `:q!`
- 在vim里面运行命令行: `:!`后带命令，可以将`:`后的`!`理解为允许vim执行外部命令。
- 在visual mode中选取多行后，可以通过`:w filename`将选取的行保存到filename当中。

### 提取、合并文件

- 在当前位置插入另外文件的内容: `:r filename`
- `:r`也可以读取外部命令的输出，比如可以通过`:r!ls`来将ls的输出放置到当前光标下面

### 复制粘贴, 替换修改

- 复制: `yy`
- 粘贴: `p`
- 使用`dd`删除的行使用`p`也可以进行粘贴
- 将当前光标下的字符替换为想要的字符: `r`后加上一个字符
- 字符`c`的功能: 更改字符
  - 使用方式: `c [number] motion`，motion则是之前常用的动作参数。使用后会自动转换到插入模式。
  - 从光标处删除两个单词并自动转换到插入模式进行修改: `c2w`
  - 从光标处删除到行尾柄并自动转换到插入模式进行修改:`c$`
  - 该变文本直到一个单词的末尾并自动转换到插入模式进行修改: `ce`

**注: 使用复制或者删除的时候复制或删除的内容会存到寄存器中，在当前vim环境下面可以进行粘贴，但是无法在vim外进行粘贴。如果想要将内容复制到剪贴板中，可以在visual mode下选中信息后输入`"+y`来将选中的信息复制到剪贴板中。注意需要通过`vim --version | grep clipboard`查看vim版本是否支持剪贴板。如果输出包含了`+clipboard`，则说明vim版本支持剪贴板。**

修改数字:

- 直接跳转到第一个数字并加1: `Ctrl-a`
- 直接跳转到第一个数字并减1: `Ctrl-x`
- 可以在前面加上数字来表示执行多次: `10Ctrl-a`执行10次加1
- vim把以0开头的数字解释为八进制值，而不是十进制。如果想要修改默认为十进制，则可以在vimrc中添加`set nrformats=`

### 定位

- 输入`Ctrl-G`即可得知当前光标所在行位置以及文件信息
- 光标跳转到最后一行: `G`
- 光标跳转到第一行: `gg`
- 跳转到指定行: `number G`
- 在normal模式下跳转到首个匹配的字符: `f + char`。例如 `fX`，表示在当前行内向后查找字符`X`，光标会移动到第一个匹配到的字符位置。使用`t + char`则跳转到首个匹配的字符的**前面一个的位置**。
  - 而如果选择向前匹配，则可以使用`F + char`和`T + char`。
  - `;`键表示为执行上一步查找的命令(f, F, t, T)。
  - `,`键则与`;`表示相反的命令。如果`;`表示前进，那么`,`表示回退。
  - 这种方式的使用非常便捷，而且易于修改比如一个长句子包含了"hello, goodbye."，我们可以使用`f,`跳转到`,`处，然后使用`dt.`将包括`,`以及后面的信息删除，除了最后的`.`

### 撤销

- 撤销: `u`
- 撤销一整行的修改: `U`
- 重写: `Ctrl-R`

### 删除

- 删除整行: `dd`
- 另一种删除方法: `c+指令`，用法与`d+指令`相同，区别在于`c`在`d`的基础上又转化为了插入模式，相当于删除后开始写入信息，不需要`d+指令`后输入`i`进入插入模式。比如`cw`就是删除当前光标到单词结尾的信息并进入插入模式。
- 删除当前字符: `x`
- 删除当前字符并进入插入模式: `s`
- 删除到行尾: `D`
- 在visual mode中选取文本内容后可以通过输入`d`删除选中的文本内容。
- 删除该单词(只要光标在该单词范围内): `daw` 可以理解为`delete a word`的缩写

### 组合快捷键

- 删除两个单词: `d2w`
- 删除单词，执行两次: `2dw`
- 删除两个单词，执行两次: `2d2w`
- 在视图模式下选中后5行删除: `d5j`

### 搜索替换

- 在当前光标下搜索下一个匹配的信息: `/` + 匹配的信息
- 在当前光标下搜索上一个匹配的信息: `?` + 匹配的信息
- 搜索之后跳转到下一个匹配的信息: `n`
- 搜索之后跳转到上一个匹配的信息: `N`
- 快速搜索当前光标的单词: 向后 `*`  向前 `#`，之后也可以使用`n`和`N`来改变方向
- 将range范围内的from替换为to： `:[range]s/from/to/[flags]`
- 在要查找的内容后面加上“\\c”（不区分大小写）或“\\C”（区分大小写），比如`/`+匹配的信息后面加上`\c`或`\C`或者在`:[range]s/from/to/[flags]`的`from`后加上`\c`或`\C`
- 还有一种搜索方式就是让光标某个单词中按下`*`，会自动跳转下一个匹配的单词。如果想要高亮匹配的单词，可以使用`:set hls`，关闭高亮可以使用`:nohls` (另一种打开和关闭高亮的方式是`:set hlsearch`和`:set nohlsearch`)。搜索一次过后也可以使用`N`和`n`。

range列表

| Range   | Description                                 | Example             |
| ------- | :------------------------------------------ | ------------------- |
| `21`    | line `21`                                   | `:21s/old/new/g`    |
| `1`     | first line                                  | `:1s/old/new/g`     |
| `$`     | last line                                   | `:$s/old/new/g`     |
| `%`     | all lines, same as `1,$`                    | `:%s/old/new/g`     |
| `21,25` | lines 21 to 25                              | `:21,25s/old/new/g` |
| `21,$`  | lines 21 to end                             | `:21,$s/old/new/g`  |
| `.,$`   | current line to end                         | `:.,$s/old/new/g`   |
| `.+1,$` | line after current line to end              | `:.+1,$s/old/new/g` |
| `.,.+5` | six lines (current to current +5 inclusive) | `:.,.+5s/old/new/g` |
| `.,.5`  | same (`.5` is intepreted as `.+5`)          | `:.,.5s/old/new/g`  |

有些特殊符号需要在前面加上`\`才能识别。

需要注意的是，如果同一行有多个能匹配到的位置，替换的话只会替换第一个匹配的信息。添加flag: g可以实现每一行中所有匹配的替换(比如上面range列表中的最后的`/g`)。

flag list

| flag | 作用                                                         |
| ---- | ------------------------------------------------------------ |
| &    | 复用上次替换命令的flags                                      |
| g    | 替换每行的所有匹配值(默认没有g的情况下只会替换每行的第一个匹配值) |
| c    | 替换前需确认                                                 |
| e    | 替换失败时不报错                                             |
| i    | 大小写不敏感                                                 |
| I    | 大小写敏感                                                   |

此外，使用`sed`可以直接将某个文件里面的某个信息替换为另一个信息: `sed -i "[range]s/from/to/[flags]" filename`就是将filename文件中的from替换为to。`-i`表示在文件内更改。否则更改结果只会在终端中打印出来。

### 分窗口

- 生成水平的窗口: `:sp`
- 生成垂直窗口: `:vsp`
- 移动到另一个窗口操作: `Ctrl-W` + `[hjkl]`

### 滚动窗口

1. `Ctrl+E` - 向下滚动窗口一行，不移动光标。
2. `Ctrl+Y` - 向上滚动窗口一行，不移动光标。
3. `Ctrl+D` - 向下滚动半个屏幕。
4. `Ctrl+U` - 向上滚动半个屏幕。
5. `Ctrl+F` - 向下滚动一个整屏幕。
6. `Ctrl+B` - 向上滚动一个整屏幕。
7. `zz` - 将当前行移至窗口中央，光标位置不变。
8. `zt` - 将当前行移至窗口顶部，光标位置不变。
9. `zb` - 将当前行移至窗口底部，光标位置不变。

### 生成标签 (便于跳转)

生成的标签可以是小写字母a-z或者大写字母a-z，也可以是数字0-9。小写字母的标记，仅用于当前缓冲区；而大写字母的标记和数字0-9的标记，则可以跨越不同的缓冲区。小写字母的标签可以被`delmarks!`删除，大写字母和0-9不行。大写字母和0-9只能通过`delmarks character`来进行删除

- 生成一个标签a: `ma`
- 跳转到标签a所在位置: ``a`
- 跳转到标签a所在的行首: `'a`
- 查找所有的标签: `:marks`
- 删除标签a: `:delmarks a`
- 删除a-z的标签: `:delmarks a-z`
- 删除A-Z的标签: `:delmarks A-Z`
- 删除所有标签(不包括大写的标签): `:delmarks!`

### 注释代码

可以使用visual block模式来注释多行代码

- visual block: `Ctrl V`
- 在visual block模式下通过[hjkl]选中多行后，使用`I`来进行插入，例如输入`//`然后`Esc`即可实现多行注释。

使用注释插件

https://github.com/tpope/vim-commentary

```
mkdir -p ~/.vim/pack/tpope/start
cd ~/.vim/pack/tpope/start
git clone https://tpope.io/vim/commentary.git
vim -u NONE -c "helptags commentary/doc" -c q
```

Use `gcc` to comment out a line (takes a count), `gc` to comment out the target of a motion.

### 代码补全

在vim中自带了基础的自动补全功能。但该功能的局限之处在于，只能补全之前已经出现过的单词。当写好了单词一部分后，输入`Ctrl-N`，自动补全功能会出现提供匹配列表、完成补全、匹配失败等三种不同的情况。

### 代码跳转

可以下载ctags来跳转到某对象的定义位置。

在代码所在路径下输入`ctags -R .`可以创建代码关联的文件tag。

默认情况下在一个代码文件里面使用关联只能在当前路径下寻找关联，在～/.vimrc里面添加`set tags=./tags;,tags`可以寻找tag文件路径下所有的位置是否有关联。

vim打开文件后，在对应的声明的地方按`Ctrl-]`就可以自动跳转到对象的定义的文件的对应位置。

直接查找某个对象(比如class\_name)的定义的文件以及对应位置: `vim -t class_name`

### 查找历史Ex命令

输入`q:`即可打开一个命令历史窗口，我们可以在此窗口中查看之前执行过的Ex命令(以冒号开头的命令)的历史记录，并且可以编辑和重新执行这些命令。编辑命令遵循vim语法。编辑完成后，按下Enter键即可执行编辑好的命令。

 

## vim相关的插件

**插件网站:** https://vimawesome.com/

**vim-plug**插件管理工具

github链接: https://github.com/junegunn/vim-plug?tab=readme-ov-file

安装教程: https://github.com/junegunn/vim-plug/wiki/tutorial

vim-plug是一个基于Rust编写的vim插件管理工具，可以轻松下载需要的vim有关的插件。

安装方式 (Unix): 

```bash
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

之后在打开`~/.vimrc`文件添加下面的信息：

```bash
" Plugins will be downloaded under the specified directory.
call plug#begin(has('nvim') ? stdpath('data') . '/plugged' : '~/.vim/plugged')

" Declare the list of plugins.
Plug 'tpope/vim-sensible'
Plug 'junegunn/seoul256.vim'

" List ends here. Plugins become visible to Vim after this call.
call plug#end()
```

之后重启vim就可以使用Plugin插件。在vim打开文件内输入`:Pluginstall`即可下载`~/.vimrc`中声明的插件。输入`:PluginUpdate`可以更新`~/.vimrc`中新添加的插件。输入`:PlugClean`可以清楚`~/.vimrc`中被删除的插件。

### fzf.vim

https://github.com/junegunn/fzf.vim

该插件包含了fzf的很多功能，并且移植到了vim中。可以使用诸如`:Ag`等功能。详情可以查看上面的源码链接。

### NERDTree

NERDTree可以在vim打开文件后在左边栏显示当前路径下的文件。

- vim打开文件后输入`:NERDTree`即可在左边栏显示当前路径下的文件信息。左边栏可以选择目录中不同文件，按`ENTER`即可显示选中的文件信息。
- 光标左右界面跳转: `Ctrl-WW`
- 在对应的vim界面命令行模式下输入退出命令，q!或wq，即可退出对应的界面。
- 在`~/.vimrc`中添加`autocmd VimEnter * NERDTree`即可在vim打开文件后自动开启NERDTree插件。

### EasyComplete

https://zhuanlan.zhihu.com/p/366496399

超轻量级的vim代码补全工具，如果想要更全面的补全功能，可以尝试coc

### vim-colors-solarized

https://vimawesome.com/plugin/vim-colors-solarized-ours

vim 文本编辑器的精确配色方案

