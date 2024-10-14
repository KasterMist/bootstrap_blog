---
# type: docs 
title: Vim用法 (四) 代码跳转详解
date: 2024-06-14T09:31:02+08:00
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

使用vim在处理简单的代码文件的时候非常方便。但是如果想要处理一个大型项目的话，则较为麻烦。因为浏览大型项目需要不断跳转代码，所以这篇博客将介绍如何使用vim进行代码跳转。

<!--more-->

## 使用ctags配置标签文件

使用`ctags`可以创建代码库的索引。`ctags`需要自行下载。在当前工作路径下调用`ctags -R .`(`.`表示当前路径，可以省略)即可创建在tags路径下创建标签文件。

### 更新配置文件

这个配置文件是静态的，如果项目代码修改了话，我们也需要实时修改tag文件。我们可以通过使用快捷键映射来进行tag更新，比如在`~/.vimrc`中添加`nnoremap <f5> :!ctags -R<CR>`即可在vim打开的情况下按`<f5>`更新配置文件。

我们也可以编辑vim的自动命令(autocommand)功能允许我们在某个事件发生时调用一条命令，可以在`~/.vimrc`中创建下面这条命令来每次保存文件的时候自动调用ctags:

```bash
autocmd BufWritePost * call system("ctags -R")
```

不过这种方式会让所有vim打开的文件再保存时都会更新(创建)ctags，如果我们不想对某个文件进行ctags配置标签，这种方式就不太可行了。我们可以自定义函数来识别某个文件是否之前已经包含在了ctags标签文件里面，如果在的话，就进行更新ctags。而因为命令是在`~/.vimrc`中，我们需要获取保存文件的路径来替换`ctags -R`的路径，不然每次创建ctags就会一直在`~/.vimrc`所在路径当中。

下面是编写的`~/.vimrc`:

```bash
" 设置autocmd来在保存文件时自动更新ctags
autocmd BufWritePost * call UpdateCtags()

" 定义UpdateCtags函数，让之前设置过ctags的文件更新ctags，没有ctags的文件并不会创建ctags
function! UpdateCtags()
	" 获取当前文件路径
    let l:dir = expand('%:p:h')
    if filereadable(l:dir . '/tags')
		" 在当前文件所在目录更新tags文件
        execute 'silent! !ctags -R ' . l:dir
		" 排序tag文件
        execute 'silent! !sort ' . l:dir . '/tags -o ' . l:dir . '/tags'
        echo "Ctags updated and sorted"
    endif
endfunction
```



## 关键字跳转

在配置好ctags后，我们可以在当前光标所在关键字下输入`<Ctrl-]>`，这样就会自动跳转到关键字第一次声明的地方。相同的Ex语句是`:tag <keyword>`，我们可以通过`:tag <keyword>`直接输入某个关键字来跳转。使用`<Ctrl-t>`会充当后退按钮，跳回到keywords的地方。

### 新建标签页进行跳转

不过，如果我们想要实现像一般IDE那样多个标签页跳转的话，可以通过`:tab tag <keyword>`的方式来实现。如果想要在当前光标下不输入keyword就能实现的话，可以调用`tab tag <Ctrl-r><Ctrl-w>`，这样tag后面会补全光标下的keyword内容。

我们也可以使用快捷键来新建标签页进行跳转。不过需要注意的是，如果当前标签页已经打开了对应的文件，vim是检测不到的，会新建一个新的标签页再次打开对应的文件，我们可以通过编写函数来自定义查找当前标签页，如果已经有对应的文件，则直接跳转到对应文件的标签页，没有的话就再创建标签页。

下面是在`~/.vimrc`的实现方式：

```bash
function! OpenTagInNewTab()
    let tag_name = expand('<cword>')
    " 获取标签定义位置
    let tag_info = taglist(tag_name)
    if empty(tag_info)
        echo "Tag not found"
        return
    endif
    let file_name = tag_info[0].filename

    " 遍历所有标签页，查找包含目标文件的标签页
    let found = 0
    for i in range(1, tabpagenr('$'))
        execute i . 'tabnext'
        if bufname('%') == file_name
            execute 'tag ' . tag_name
            let found = 1
            break
        endif
    endfor

    " 如果未找到，打开新标签页
    if !found
        execute 'tabedit ' . file_name
        execute 'tag ' . tag_name
    endif
endfunction

nnoremap <Leader> :call OpenTagInNewTab()<CR>
```

我们设置了快捷键`<leader>`(\\按键)，使用这个按键即可调用自定义的函数。

注: `~/.vimrc`中的命令都是使用Vimscript编写的。Vimscript是一种特定于 Vim 编辑器的脚本语言，用于编写vim配置文件(.vimrc)和插件。如果想要自定义更多的功能，需要系统性的学习Vimscript语言。



## 在文件间跳转



### 遍历跳转列表

vim可以通过快捷键在文件内跳转行以及文件之间跳转行。我们需要知道vim中的对跳转命令的定义。在vim中，**任何改变当前窗口活动文件的命令都可以被称为跳转命令**。此外，使用类似`[count]G`命令直接跳到指定的行号也会被当成一次跳转，而每次向上向下移动一行不算。可以理解为，大范围的动作命令可能会被当成跳转，小范围的动作命令会被当作成移动。

使用`:jumps`可以查看跳转列表的内容。

使用`<C-o>`(向后)和`<C-i>`(向前)可以根据跳转列表的内容在新文件以及旧文件之间进行跳转。



### 遍历改变列表

vim在编辑期间回维护一张表，里面记载着对每个缓冲区所做的修改。

使用`:changes`可以查看改变列表的内容。

我们可以使用`g;`和`g,`来反向或正向遍历改变列表 (改变列表里面的改变从上到下是由旧到新排列，但是index则是从大到小，使用`g;`则是index递增遍历，即往前遍历改变列表)。



### 生成标签便于跳转

可参考vim用法(一)中的生成标签功能
