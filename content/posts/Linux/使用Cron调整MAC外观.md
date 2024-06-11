---
# type: docs 
title: 使用Cron调整MAC外观
date: 2024-06-11T16:41:55+08:00
featured: false
draft: false
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Note"]
tags: ["macOS Usage"]
images: []
---

本人使用macbook一般都会设置护眼模式，即夜览模式。然而，macbook会认为夜览模式就是在晚上才会启用。当设置为永久夜览模式后，如果将外观模式设置为自动，那么macbook会默认永久使用深色外观。然而我想要实现白天为浅色外观，晚上为深色外观。这篇文章将介绍如何使用Cron调度来自动更新MAC的浅色外观和深色外观。

<!--more-->

在之前的Linux常用命令博客中已经提到了Cron是一个很好用的设置定时任务的命令。我们可以创建脚本来切换外观，然后通过Crontab来执行定时任务调用切换外观的脚本。

## 创建脚本

首先创建切换为浅色外观和深色外观的脚本。

深色模式脚本 `dark_mode.applescript`

```bash
tell application "System Events"
    tell appearance preferences
        set dark mode to true
    end tell
end tell
```

浅色模式脚本`light_mode.applescript`

```bash
tell application "System Events"
    tell appearance preferences
        set dark mode to false
    end tell
end tell
```

然后创建一个shell脚本`toggle_dark_light.sh`来调用这两个脚本，同时加入一个参数来选择调用哪一个脚本:

```bash
#!/bin/bash

if [ "$1" == "dark" ]; then
    osascript /path/to/dark_mode.applescript
elif [ "$1" == "light" ]; then
    osascript /path/to/light_mode.applescript
else
    echo "Invalid argument: use 'dark' or 'light'"
fi
```

(将path切换为自己的path)

## 配置cron作业

使用 `crontab -e` 编辑 `cron` 表，可以添加下面的代码:

```bash
# 每天早上7点切换到浅色模式
0 7 * * * /path/to/toggle_dark_light.sh light

# 每天晚上7点切换到深色模式
0 19 * * * /path/to/toggle_dark_light.sh dark
```

(将path切换为自己的path)

保存后，Macbook将会启用cron作业来根据时间执行不同的任务。
