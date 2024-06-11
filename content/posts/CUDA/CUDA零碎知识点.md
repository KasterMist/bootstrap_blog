---
# type: docs 
title: CUDA零碎知识点
date: 2024-06-06T09:42:20+08:00
featured: false
draft: true
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Tutorial"]
tags: ["CUDA"]
images: []
---

在学习CUDA过程中会遇到很多问题，而这些问题过于零碎，无法进行系统化的分类，所以这篇文章主要记录平时遇到的CUDA使用的知识点。

<!--more-->



#### thrust

thrust是C++的一个扩展库，其中thrust::device\_vector用于在设备（GPU）内存中存储数据。我们可以在host中通过创建thrust::device\_vector来创建一个存储在device的vector并赋值。下面是一个例子:

```c
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main() {
    // 创建并初始化 host_vector
    thrust::host_vector<int> h_vec(5);
    h_vec[0] = 1; h_vec[1] = 2; h_vec[2] = 3; h_vec[3] = 4; h_vec[4] = 5;

    // 将数据从 host_vector 复制到 device_vector
    thrust::device_vector<int> d_vec = h_vec;

    // 可以在 CPU 上通过 host_vector 访问 device_vector 的内容
    thrust::host_vector<int> h_vec_copy = d_vec;
    for (int i = 0; i < h_vec_copy.size(); i++) {
        std::cout << h_vec_copy[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

此外，我们也可以直接对device\_vector进行赋值，比如`thrust::device_vector<int> d_vec = {1, 1, 1};`，但是在host端我们仍然无法直接访问device\_vector的值。

使用thrust的拓展可以简化CUDA编程，尤其是内存的管理和数据传输。

thrust::copy用于在GPU和CPU之间复制数据。下面的例子是具体用法:

```c
int main() {
    // 创建主机向量并初始化
    thrust::host_vector<int> h_vec(5);
    h_vec[0] = 10; h_vec[1] = 20; h_vec[2] = 30; h_vec[3] = 40; h_vec[4] = 50;

    // 创建设备向量并分配内存
    thrust::device_vector<int> d_vec(5);

    // 从主机向量复制到设备向量
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

    // 从设备向量复制回主机向量
    thrust::host_vector<int> h_vec_copy(5);
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec_copy.begin());

    // 打印结果
    for(int i = 0; i < h_vec_copy.size(); i++) {
        std::cout << h_vec_copy[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

