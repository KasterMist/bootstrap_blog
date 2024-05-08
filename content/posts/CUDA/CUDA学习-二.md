---
title:       	"CUDA学习(二)"
subtitle:    	""
description: 	""
date:        	2024-02-08
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

这一部分将介绍CUDA的并行编程方式

<!-- more -->



## 矢量求和运算

假设有两组数据，我们需要将这两组数据中对应的元素两两相加，并将结果保存在第三个数组中。



### 基于CPU的矢量求和

首先，下面的代码是通过传统的C代码来实现这个求和运算

```c
#include "../common/book.h"

#define N 10

void add(int *a, int *b, int *c){
	int tid = 0;	//这是第0个CPU，因此索引从0开始
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += 1;	//由于只有一个CPU，因此每次递增1
    }
}

int main(){
    int a[N], b[N], c[N];
    
    //在CPU上为数组"a"和"b"赋值
    for(int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i * i;
    }
    
    add(a, b, c);
    
    //显示结果
    for(int i = 0; i < N; i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    
    return 0;
}
```

add()中使用while循环而不是for循环是为了代码能够在拥有多个CPU或者多个CPU核的系统上并行运行，比如双核处理器上可以将每次递增的大小改为2。

```c
//第一个CPU核
void add(int *a, int *b, int *c){
    int tid = 0;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += 2;
    }
}

//第二个CPU核
void add(int *a, int *b, int *c){
    int tid = 1;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += 2;
    }
}
```



### 基于GPU的矢量求和

下面是基于GPU的矢量求和代码

```c
#include "../common/book.h"

#define N 10

__global__ add(int *dev_a, int *dev_c, int *dev_c){
    int tid = blockIdx.x; //计算该索引处的数据
    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

int main(){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
    
    //在CPU上为数组"a"和"b"赋值
    for(int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i * i;
    }
    
    //将数组"a"和"b"复制到GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    
    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    
    //将数组"c"从GPU复制到CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    //显示结果
    for(int i = 0; i < N; i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    
    //释放在GPU上分配的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    return 0;
}
```

在示例代码中，调用add函数的尖括号内的数值是<<<N, 1>>>，其中第一个参数表示设备在执行核函数时使用的并行线程块的数量。比如如果制定的事kernel<<<256, 1>>>()，那么将有256个线程块在GPU上运行。

在add函数里面，我们可以使用blockIdx.x获取具体的线程块(blockIdx是一个内置变量，不需要定义它)，通过这种方式可以让不同的线程块并行执行数组的矢量相加。

下一章将会详细解释线程块以及线程之间的通信机制和同步机制。
