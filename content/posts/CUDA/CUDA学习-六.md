---
title:       	"CUDA学习(六)"
subtitle:    	""
description: 	""
date:        	2024-02-27
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

在某些情况中，对于单线程应用程序来说非常简单的任务，在大规模并行架构上实现时会变成一个复杂的问题。在本章中，我们将举例其中一些情况，并在这些情况中安全地完成传统单线程应用程序中的简单任务。

<!-- more -->



## 原子操作简介

在编写传统的单线程应用程序时，程序员通常不需要使用原子操作。下面会介绍一下原子操作是什么，以及为什么在多线程程序中需要使用它们。我们先分析C或者C++的递增运算符：

x++;

在这个操作中包含了三个步骤：

1. 读取x中的值。
2. 将步骤1中读到的值增加1。
3. 将递增后的结果写回到x。

有时候，这个过程也称为读取-修改-写入操作。

当两个线程都需要对x的值进行递增时，假设x的初始值为7，理想情况下，两个线程按顺序对x进行递增，第一个线程完成三个步骤后第二个线程紧接着完成三个步骤，最后得到的结果是9。但是，实际情况下会出现两个线程的操作彼此交叉进行，这种情况下得到的结果将小于9(比如两个线程同时读取x=7，计算完后写入，那样的话x最后会等于8)。

因此，我们需要通过某种方式一次性地执行完读取-修改-写入这三个操作，并在执行过程中不会被其他线程中断。**由于这些操作的执行过程不能分解为更小的部分，因此我们将满足这种条件限制的操作称为原子操作。**



## 计算直方图

本章将通过给出计算直方图的例子来介绍如何进行原子性计算。

### 在CPU上计算直方图

某个数据的直方图表示每个元素出现的频率。在示例中，这个数据将是随机生成的字节流。我们可以通过工具函数big_random_block()来生成这个随机的字节流。在应用程序中将生成100MB的随机数据。

```c
#include "../common/book.h"

#define SIZE (100 * 1024 * 1024)

int main(){
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
}
```

由于每个随机字节（8比特）都有256个不同的可能取值（从0x00到0xFF），因此在直方图中需要包含256个元素，每个元素记录相应的值在数据流中的出现次数。我们创建了一个包含256个元素的数组，并将所有元素的值初始化为0。

```c
unsigned int histo[256];
for(int i = 0; i < 256; i++){
    histo[i] = 0;
}
```

接下来需要计算每个值在buffer[]数据中的出现频率。算法思想是，每当在数组buffer[]中出现某个值z时，就递增直方图数组中索引为z的元素，这样就能计算出值z的出现次数。如果当前看到的值为buffer[i]，那么将递增数组中索引等于buffer[i]的元素。由于元素buffer[i]位于histo[buffer[i]]，我们只需一行代码就可以递增相应的计数器。在一个for循环中对buffer[]中的每个元素执行这个操作：

```c
for(int i = 0; i < SIZE; i++){
    histo[buffer[i]]++;
}
```

接下来将验证直方图的所有元素相加起来是否等于正确的值。

```c
long histoCount = 0;
for(int i = 0; i < 256; i++){
    histoCount += histo[i];
}
printf("Histogram Sum: %ld\n", histoCount);

free(buffer);
```



### 在GPU上计算直方图

以下时计算直方图的GPU版本

```c
int main(){
    unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
    
    //初始化计时事件
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    //在GPU上为文件的数据分配内存
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    HANDLE_ERROR(cudaMallc((void**)&dev_buffer, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
    
    HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));
    
    
    unsigned int histo[256];
    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    
    //得到停止事件并显示计时结果
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    
    //下面是验证直方图的总和是否等于正确的值，因为是在CPU上运行，并不需要对此进行计时
    long histoCount = 0;
    for(int i = 0; i < 256; i++){
        histoCount += histo[i];
    }
    printf("Histogram Sum: %ld\n", histoCount);
    
    //验证GPU与CPU的搭配的是相同的计数值
    for(int i = 0; i < SIZE; i++){
        histo[buffer[i]]--;
    }
    for(int i = 0; i < 256; i++){
        if(histo[i] != 0){
            printf("Failure at %d!\n", i);
        }
    }
    
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);
}
```

cudaMemset()与C的memset()是相似的，不同之处在于cudaMemset()将返回一个错误码，将高速调用者在设置GPU内存时发生的错误。

接下来会介绍GPU上计算直方图的代码。计算直方图的核函数需要的参数包括：

- 一个指向输入数组的指针
- 输入数组的长度
- 一个指向输出直方图的指针

核函数执行的第一个计算就是计算输入数据数组中的偏移。每个线程的起始偏移都是0到线程数量减1之间的某个值，然后，对偏移的增量为已启动线程的总数。

```c
#include "../common/book.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    //当每个线程知道它的起始偏移i以及递增的数量，这段代码将遍历输入数组，并递增直方图中相应元素的值
    while(i < size){
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }
}
```

函数调用atomicAdd(addr, y)将生成一个原子的操作序列，这个操作序列包括读取地址addr处的值，将y增加到这个值，以及将结果保存回地址addr。底层硬件将确保当执行这些操作时，其他任何线程都不会读取或写入地址addr上的值。



然而，原子操作回导致性能降低，但是解决问题的方法有时会需要更多而非更少的原子操作。**这里的主要问题并非在于使用了过多的原子操作，而是有数千个线程在少量的内存地址上发生竞争。**要解决这个问题，我们需要将直方图计算分为两个阶段。

- 第一个阶段，每个并行线程块将计算它所处理数据的直方图。每个线程块在执行这个操作时都是相互独立的，因此可以在共享内存中计算这些直方图，这将避免每次将写入操作从芯片发送到DRAM。但是这种方式仍然需要原子操作，因为线程块中的多个线程之间仍然会处理相同值的数据元素。**不过，现在只有256个线程在256个地址上发生竞争，这将极大地减少在使用全局内存时在数千个线程之间发生竞争的情况。**我们将使用共享内存缓冲区temp[]而不是全局内存缓冲区histo[]，而且需要随后调用__syncthreads()来确保提交最后的写入操作。
- 第二个阶段则是将每个线程块的临时直方图合并到全局缓冲区histo[]中。由于我们使用了256个线程，并且直方图中包含了256个元素，因此每个线程将自动把它计算得到的元素只增加到最终直方图的元素上（如果线程数量不等于元素数量，那么这个阶段将更为复杂）。我们并不保证线程块将按照何种顺序将各自的值相加到最终直方图中，但由于整数加法时可交换的，无论哪种顺序都会得到相同的结果。

```c
__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int *histo){
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(i < size){
        atomicAdd(&temp[buffer[i]], 1);
        i += offset;
    }
    __syncthreads();
    atomicAdd(*(histo[threadIdx.x]), temp[threadIdx.x]);
    
}
```

