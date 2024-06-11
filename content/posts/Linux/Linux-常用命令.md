---
title:			"Linux 常用命令"
subtitle:		""
description: 	""
date:        	2024-04-04
#type:   		docs
featured:    	false
draft:    		false
comment:    	true
toc:    		true
reward:    		true
pinned:    		false
carousel:    	false
series:
categories:   	["Note"]
tags:    		["Linux", "Linux Command"]
images:    		[]
---

## 基础命令



#### 查询文件和子目录 ls

查询文件和子目录的最简单的命令是`ls`。它可以列出当前目录的文件和子目录。常用的指令有:

- `ls /path/to/directory` 列出指定目录中的文件和子目录
- `ls -a`列出隐藏文件和子目录
- `ls -l`以详细格式列出文件和子目录，包含读写权限、创建时间等信息

<!--more-->



#### 查询当前工作目录的绝对路径 pwd

查询当前工作目录的绝对路径是`pwd`



#### 改变当前工作目录 cd

使用`cd`可更改当前工作目录，可以在当前目录下使用`cd subpath`进入子目录，也可以使用`cd absolutePath`进入绝对路径目录下。

使用`cd ..`可移动到上一级目录。



#### 创建目录 mkdir

`mkdir` 命令用于创建一个新的目录。

- `mkdir directory`在当前目录下创建一个directory的子目录
- `mkdir -p /path/to/new/directory`如果要递归创建目录的话，需要加上`-p`flag



#### 删除目录 rmdir

`rmdir`只能用于删除空的目录，如果是非空目录，则需要时用`rm`

- `rmdir -v /path/subpath`会删除path里面的subpath子目录，同时显示详细的输出。
- `rmdir -p /path/subpath`会首先删除path里面的subpath子目录，之后会尝试删除path目录。



#### 创建空文件 touch

- `touch filename`创建一个名叫filename的空文件。
- `touch -c filename`如果文件已经存在，则不创建该文件。这样可以避免意外覆盖现有文件。



#### 复制文件和目录 cp

语法: `cp [option] source destination`

- `cp old_path/old_file.txt new_path/new_file.txt`将old\_path的old\_file.txt以new\_file.txt的名字复制到new\_path中。如果不更改名字的话，destination可以只写`new_path`
- `cp -r old_path/old_sub_path new_path/new_subpath`递归复制整个目录



#### 移动或者重命名文件和目录 mv

语法: `mv [options] source destination`

- `mv old_path/old_file.txt new_path/new_file.txt`将old\_path的old\_file.txt以new\_file.txt的名字移动到new\_path中。
- `mv -i old_path/old_file.txt new_path/new_file.txt`在覆盖目标位置的任何现有文件前提示。这样可以防止意外覆盖数据。
- `mv old_file.txt new_file.txt`可以通过mv在同一路径下重命名old\_file.txt为new\_file.txt。



#### 移除文件和目录 rm

rm在使用时需要格外小心，因为恢复使用rm删除的文件和目录会非常困难。

`rm -r name`递归删除目录，包括目录里面的所有内容。

`rm -f name`强制删除并抑制所有提示。

`rm -i name`在删除每个文件或目录前提示确认，以防意外删除。

比如使用`rm -rf path`可以强制删除path路径和path里面的所有内容。



#### 在目录层次结构中搜索文件 find

语法: `find [path] [critical]`

示例

- `find . -name example.txt`: 查找当前目录及其子目录中所有名为 `example.txt` 的文件
- `find /home -type f`: 查找 `/home` 目录中所有的普通文件
- `find / -type f -size +1M`: 查找文件大小大于 1MB 的文件
- `find / -name name`: 在`/`路径下查找名字为`name`的文件



#### 使用条件匹配搜索文本 grep

语法: `grep [options] pattern [files]`

示例

- `grep "a" example.txt`搜索`example.txt`中"a"这个单词。
- `grep -c "a" example.txt`搜索`example.txt`中"a"这个单词出现的次数。



#### 比较文件 diff

语法: `diff [options] file1 file2`

用于比较两个文件的差异

`diff original.txt updated.txt`比较 original.txt和updated.txt两个文件的差异，会输出产生差异的不同行。



#### 字数统计 wc

`wc -l example.txt`: 只打印行计数

`wc -w example.txt`: 只打印字数

`wc -c example.txt`: 只打印字节数

#### 历史命令 history

使用`history`可以查看之前输入过的命令语句。每个命令前面都有一个编号。你可以使用 `!` 加上命令编号来执行历史记录中的命令，比如`!123`来执行第123号的命令。还可以使用 `!!` 来执行最后一个命令，或者使用 `!string` 来执行最近包含字符串 `string` 的命令。

使用 `history -c` 命令来清除命令历史记录。



## 文件权限命令



#### 更改文件模式或访问权限 chmod

文件的权限包括：只读(r)，写入(w)，执行(x)。

模式：指定要修改的权限模式。权限模式可以使用数字表示，也可以使用符号表示。

- 数字表示：使用三位数字（0-7）表示权限。每一位数字代表一个权限位，分别对应于读（4）、写（2）和执行（1）权限。例如，755 表示所有者具有读、写和执行权限，组和其他用户具有读和执行权限。
- 符号表示：使用符号来表示权限。符号表示包括以下几个部分：
  - 操作符：可以是 `+`（添加权限）、`-`（删除权限）或 `=`（设置权限）。
  - 权限范围：可以是 `u`（所有者）、`g`（组）、`o`（其他用户）或 `a`（所有用户）。
  - 权限类型：可以是 `r`（读取权限）、`w`（写入权限）或 `x`（执行权限）。

例如，`u+x` 表示为所有者添加执行权限，`go-w` 表示删除组和其他用户的写入权限。

示例:

- `chmod 644 example.txt`: 将文件 `example.txt` 的权限设置为所有者可读写、组和其他用户只读。
- `chmod -R 777 documents`: 将目录 `documents` 及其子目录中所有文件的权限设置为所有者可读写执行，组和其他用户可读写执行 (-R用于递归地修改目录及其子目录中的文件权限)。
- `chmod +x script.sh`: 为 `script.sh` 添加执行权限。



## 管理命令



#### 查看当前进程信息 ps

用于显示当前正在运行的进程信息。它可以显示当前用户的进程、所有用户的进程或者系统的所有进程。



#### 显示linux进程 top

动态显示系统的进程信息和资源使用情况。它会实时更新显示当前正在运行的进程列表，并且会以交互式的方式展示系统的 CPU 使用情况、内存使用情况等



#### 交互式进程浏览 htop

`htop` 命令是 `top` 命令的改进版本，提供了更加友好和直观的界面，并且支持更多的交互操作。它可以显示更多的进程信息，并且可以通过键盘快捷键进行排序、过滤、查找等操作。



#### 向进程发送终结信号 kill

`kill PID`: 通过输入PID(进程ID)或程序的二进制名称来终结进程。

`kill -9 name`: 通过输入进程名称来终结进程，需要添加`-9`选项。

示例: 查找一个进程并终结

1. `ps aux | grep example_process`: 使用 `ps` 命令查找名为 `example_process` 的进程，述命令会显示包含 `example_process` 关键词的进程信息，并输出其 PID。
2. `kill PID`: 在获取到PID后，即可通过`kill PID`的方式来终结进程。



#### 报告虚拟内存统计数据 vmstat

打印有关内存、交换、I/O 和 CPU 活动的详细报告。其中包括已用/可用内存、交换入/出、磁盘块读/写和 CPU 进程/闲置时间等指标。

- `vmstat -n 5`: 每隔5秒更新一次信息。
- `vmstat -a`: 显示活动和非活动内存。
- `vmstat -s`: 显示事件计数器和内存统计信息。
- `vmstat -S`: 以 KB 而不是块为单位输出。



#### 报告CPU和I/O统计数据 iostat

监控并显示 CPU 利用率和磁盘 I/O 指标。其中包括 CPU 负载、IOPS、读/写吞吐量等。

- `iostat -c`: 显示CPU使用率信息。
- `iostat -t`: 为每份报告打印时间戳。
- `iostat -x`: 显示服务时间和等待计数等扩展统计信息。
- `iostat -d`: 显示每个磁盘/分区的详细统计信息，而不是合计总数。
- `iostat -p`: 显示特定磁盘设备的统计信息。



#### 显示可用和已用内存量 free

- `free -b`: 以字节为单位显示输出。
- `free -k`: 以KB为单位显示输出结果。
- `free -m`: 以MB为单位显示输出，而不是以字节为单位。
- `free -h`: 以GB、MB等人类可读格式打印统计数据，而不是字节。



#### 自动化

##### cron

`cron` 是一个用于设置定时任务的命令。

cron 的功能

- **定时任务调度**：根据预定的时间表执行任务。
- **后台运行**：在后台持续运行，无需用户干预。
- **灵活调度**：支持多种时间格式，如分钟、小时、天、月和星期几。

**crontab** 是管理 `cron` 作业的工具，提供了编辑、查看和删除 `cron` 作业的方法。`crontab` 文件包含了 `cron` 作业的配置。

crontab 的功能

- **编辑定时任务**：使用 `crontab -e` 命令打开并编辑 `crontab` 文件。
- **查看定时任务**：使用 `crontab -l` 命令列出当前用户的 `crontab` 文件内容。
- **删除定时任务**：使用 `crontab -r` 命令删除当前用户的 `crontab` 文件。

**crontab 文件语法**

`crontab` 文件中的每一行代表一个定时任务，格式如下：

```bash
* * * * * command_to_execute
- - - - -
| | | | |
| | | | +----- 一周中的第几天 (0 - 7) (0 和 7 都表示星期天)
| | | +------- 一个月中的第几天 (1 - 31)
| | +--------- 月份 (1 - 12)
| +----------- 小时 (0 - 23)
+------------- 分钟 (0 - 59)
```

保存crontab文件后即可自动开始任务调度。



## 有用的Unix插件

#### fzf

参考资料:https://zhuanlan.zhihu.com/p/41859976

github源码: https://github.com/junegunn/fzf

fzf是一种非常好用的下拉查找工具，通常需要与其他的命令组合。下面是一些常用的功能:

- 单独使用`fzf`命令会展示当前目录下所有文件列表，可以用键盘上下键或者鼠标点出来选择。
- 使用vim组合fzf来查找并打开当前目录下的文件: `vim $(fzf)`
- 切换当前工作目录: `cd $(find * -type d | fzf)`，其实现逻辑如下:
  1. 使用find命令找出所有的子目录
  2. 把子目录列表pipe到fzf上进行选择
  3. 再把结果以子命令的形式传给cd
- 可以将`cmd | fzf`理解为将列出的结果以fzf下拉查找工具的方式来实现，比如`ls | fzf`就是会通过下拉查找的方式查看当前路径下(不包括子路径)的文件和文件夹。`$(fzf)`意味着通过fzf选取的信息将输入到变量当中，比如`vim $(fzf)`就是通过fzf选取文件后再用vim打开。
- 切换git分支: `git checkout $(git branch -r | fzf)`

##### 使用fzf插件补全shell命令

fzf自带一种插件可以通过输入\*\*来自动生成下拉框窗口来补全信息。比如使用`cd **`然后`tab`可以使用下拉菜单选择路径。可以在https://github.com/junegunn/fzf搜索“Fuzzy completion for bash and zsh”找到有关的信息。

如果用的是homebrew，则先通过`brew install fzf`，然后执行`$(brew --prefix)/opt/fzf/install`。之后在`~/.zshrc`的`plugins=(...)`中添加`fzf`。之后执行`source ~/.zshrc`，即可使用。

我们也可以对FZF_DEFAULT_OPTS进行设置，来自定义fzf的界面，比如`FZF_DEFAULT_OPTS="--height 40% --layout=reverse --preview '(highlight -O ansi {} || cat {}) 2> /dev/null | head -500'"`



#### ag

一个类似于 `grep` 的代码搜索工具，比`grep`有更高的性能。`ag` 在搜索时会自动忽略 `.gitignore` 中的文件和目录，从而提高搜索效率。它支持正则表达式，高亮显示匹配结果，并且可以直接在你的编辑器中使用。

- 在当前目录以及子目录搜索文本: `ag "search pattern"`
- 在特定文件类型中搜索: `ag "search pattern" --cpp`只在HTML文件中搜索
- 忽略特定文件或目录: `ag "search pattern" --ignore dir/*`忽略特定目录下的搜索 `ag "search pattern" --ignore *.log` 忽略所有.log文件
- 搜索特定目录: `ag "search pattern" /path/to/directory`
- 仅显示文件名: `ag "search pattern" -l`
- 与grep一样，可以配合其他语句执行，比如`cat "filename" | ag "search pattern"`
- 引号可加可不加
