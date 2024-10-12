---
# type: docs 
title: 在Vscode中使用Cmake Tools进行Debug调试
date: 2024-10-12T11:40:40+08:00
featured: false
draft: false
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Note"]
tags: ["VsCode"]
images: ["images/logo/vscode.jpeg"]
---

这部分将介绍如何在VsCode中使用Cmake Tools进行Debug调试。



<!--more-->

我们可以通过配置tasks.json和launch.json文件来进行code run和debug配置。不过这种方式对于处理大型project来说较为麻烦，因为需要将Project的CMakeLists中的大量配置放到tasks.json和launch.json里面。

VsCode给了官方文档可供查阅: https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/debug-launch.md

但是如果想要使用CMake Tools进行Debug并自定义程序参数和Debug参数的话，需要使用launch.json。不过对于“Debug using a launch.json file”，本人按照tutorial进行了launch.json配置，但无法实现Debug时添加参数的情况。

后面发现可能是launch.json里面的设置无法对cmake.debugConfig里面进行配置，所以我采取了了在setting.json里面直接修改cmake.debugConfig里面的参数 (比如下面的代码)，之后证明可行。从VsCode的另一部分的文档中https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-settings.md#command-substitution 可以知道CMake Tools的参数配置可以在setting.json中进行设置与修改。

```json
"cmake.debugConfig": {
        "args": ["--arg", "argument info"],        // 在这里设置命令行参数
        "stopAtEntry": false
}
```



(注，如果想要实现完全的Debug功能的话，如成功设置断点，需要在CMakeLists里面对调试模式条件下加上"-g"和非高优化flag，比如"-O0"，这并不是一个难点，但是对于接手一个项目来说，我们无法判断接手的项目是否有那么完备的配置)。
