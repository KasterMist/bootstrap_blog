---
title:			"CUDA学习(一)"
subtitle:		""
description: 	""
date:        	2024-02-04
#type:		 	docs
featured: 	 	false
draft: 		 	false
comment: 	 	true
toc: 		 	true
reward: 	 	true
pinned: 	 	false
carousel: 	 	false
series:
categories:  	["Tutorial"]
tags: 		 	["CUDA"]
images: 	 	[]
---



**参考书目:** GPU高性能编程CUDA实战

**书目网页链接:** https://hpc.pku.edu.cn/docs/20170829223652566150.pdf

该博客参考于上述书籍，虽然书有一点老，但是作为初学者而言仍然能学到很多东西。

本书所包含的代码都在下面的连接中，可以下载来学习: https://developer.nvidia.com/cuda-example

<!-- more -->

## CUDA C简介

首先来看一个CUDA C的示例:

```c
#include "../common/book.h"

int main(){
  prinf("Hello World!\n");
  return 0;
}
```

这个示例只是为了说明，CUDA C与熟悉的标准C在很大程度上是没有区别的。



### 核函数调用

在GPU设备上执行的函数通常称为核函数(Kernel)

```c
#include <iostream>

__global__ void kernel(){
  
}

int main(){
  	kernel<<<1, 1>>>();
  	printf("Hello World!\n");
  	return 0;
}
```

跟之前的代码相比多了两处

- 一个空的函数kernel()，并且带有修饰符\_\_global\_\_。
- 对这个空函数的调用，并且带有修饰字符<<<1, 1>>>。

这个\_\_global\_\_可以认为是告诉编译器，函数应该编译为在设备而不是在主机上运行。函数kernel()将被交给编译器设备代码的编译器，而main()函数将被交给主机编译器。



### 传递参数

以下是对上述代码的进一步修改，可以实现将参数传递给核函数

```c
#include <iostream>
#include "book.h"

__global__ void add(int a, int b, int* c){
    *c = a + b;
    printf("c is %d\n", *c);
}

int main(void){
    int c = 0;
    int* dev_c;

    printf("original c is %d\n", c);

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));


    add<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dev_c);
    printf("2 + 7 = %d\n", c);
    return 0;
}
```

其中"book.h"包含了HANDLE_ERROR，也可以不使用"book.h"而是在代码中添加HANDLE_ERROR函数。

```c
#include <iostream>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void add(int a, int b, int* c){
    *c = a + b;
    printf("c is %d\n", *c);
}

int main(void){
    int c = 0;
    int* dev_c;

    printf("original c is %d\n", c);

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));


    add<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
		
  	printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);
    
    return 0;
}
```

- cudaMalloc()用来分配内存，这个函数的调用行为非常类似于标准C函数的malloc()，但该函数作用是告诉CUDA运行时在设备上分配内存。
  - 第一个参数是一个指针，指向用于保存新分配内存地址的变量。第二个参数是分配内存的大小。
  - 该函数返回的类型是void*。
  - 不能使用标准C的free()函数来释放cudaMallocc()分配的内存。要释放cudaMalloc()分配的内存，需要调用cudaFree()。
- HANDLE_ERROR()是定义的一个宏，作为辅助代码的一部分，用来判断函数调用是否返回了一个错误值，如果是的话，将输出相应的错误消息。
- 在主机代码中可以通过调用cudaMemcpy()来访问设备上的内存。
  - 第一个参数是目标(target)指针，第二个参数是源(source)指针，第三个参数分配内存大小。第四个参数则是指定设备内存指针。
  - 第四个参数一般有cudaMemcpyDeviceToHost，cudaMemcpyHostToDevice, cudaMemcpyDeviceToDevice三种。cudaMemcpyDeviceToHost说明我们将设备内存指针的数据传递给主机内存指针，此时第一个参数指针是在主机上，第二个参数指针是在设备上。cudaMemcpyHostToDevice说明我们将主机内存指针的数据传递给设备内存指针，此时第一个参数指针是在设备上，第二个参数指针是在主机上。此外还可以通过传递参数cudaMemcpyDeviceToDevice莱高速运行时这两个指针都在设备上。如果源指针和目标指针都在主机上，则可以直接调用memcpy()函数。

### 查询设备

我们可以使用cudaGetDeviceCount()来查询设备数量(比如GPU数量)。

```c
int count;
HANDLE_ERROR(cudaGetDeviceCount(&count));
```

CUDA设备属性包含很多信息，可以在书上或者NVIDIA官方网站上查到。

