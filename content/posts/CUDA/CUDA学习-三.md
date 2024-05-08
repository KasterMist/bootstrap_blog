---
title:       	"CUDA学习(三)"
subtitle:    	""
description: 	""
date:        	2024-02-12
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

这一章将介绍线程块以及线程之间的通信机制和同步机制。

在GPU启动并行代码的实现方法是告诉CUDA运行时启动核函数的多个并行副本。我们将这些并行副本称为线程块(Block)。

CUDA运行时将把这些线程块分解为多个线程。当需要启动多个并行线程块时，只需将尖括号中的第一个参数由1改为想要启动的线程块数量。

在尖括号中，第二个参数表示CUDA运行时在每个线程块中创建的线程数量。假设尖括号中的变量为<<<N, M>>>总共启动的线程数量可以按照以下公式计算:
$$
N个线程块 * M个线程/线程块 = N*M个并行线程
$$
<!-- more -->

### 使用线程实现GPU上的矢量求和

在之前的代码中，我们才去的时调用N个线程块，每个线程块对应一个线程`add<<<N, 1>>>(dev_a, dev_b, dev_c);`。

如果我们启动N个线程，并且所有线程都在一个线程块中，则可以表示为`add<<<1, N>>>(dev_a, dev_b, dev_c);`。此外，因为只有一个线程块，我们需要通过线程索引来对数据进行索引(而不是线程块索引)，需要将`int tid = blockIdx.x;`修改为`int tid = threadIdx.x;`



### 在GPU上对更长的矢量求和

对于启动核函数时每个线程块中的线程数量，硬件也进行了限制。具体来说，最大的线程数量不能超过设备树形结构中maxThreadsPerBlock域的值。对目前的GPU来说一个线程块最多有1024个线程。如果要通过并行线程对长度大于1024的矢量进行相加的话，就需要将线程与线程块结合起来才能实现。

此时，计算索引可以表示为:

```c
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

blockDim保存的事线程块中每一维的线程数量，由于使用的事一维线程块，因此只用到blockDim.x。

此外，gridDim是二维的，而blockDim是三维的。

假如我们使用多个线程块处理N个并行线程，每个线程块处理的线程数量为128，那样可以启动N/128个线程块。然而问题在于，当N小于128时，比如127，那么N/128等于0，此时将会启动0个线程块。所以我们希望这个除法能够向上取整。我们可以不用调用 ceil()函数，而是将计算改为(N+127)/N。因此，这个例子调用核函数可以写为:

```c
add<<<(N + 127) / 128, 128>>>(dev_a, dev_b, dev_c);
```

当N不是128的整数倍时，将启动过多的线程。然而，在核函数中已经解决了这个问题。在访问输入数组和输出数组之前，必须检查线程的便宜是否位于0到N之间。

```c
if(tid < N){
	c[tid] = a[tid] + b[tid];
}
```

因此，当索引越过数组的边界时，核函数将自动停止执行计算。核函数不会对越过数组边界的内存进行读取或者写入。



### 在GPU上对任意长度的矢量求和

当矢量的长度很长时，我们可以让每一个线程执行多个矢量相加。例如

```c
__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
```

当每个线程计算完当前索引上的任务后，接着就需要对索引进行递增，其中递增的步长为线程格中正在运行的线程数量。这个数值等于每个线程块中的线程数量乘上线程格中线程块的数量，即blockDim.x * gridDim.x。



### 共享内存和同步

CUDA C编译器对共享内存中的变量与普通变量分别采取不同的处理方式。对于在GPU上启动的每个线程块，CUDA C编译器都将创建该变量的一个副本，线程块中的每个线程都共享这块内存，但线程却无法看到也不能修改其他线程块的变量副本。这就实现了一个非常好的方式，**使得一个线程块中的多个线程能够在计算上进行通信和协作**。

而且，共享内存缓冲区驻留在物理GPU上，而不是驻留在GPU之外的系统内存中。因此，**在访问共享内存时的延迟要远远低于访问普通缓冲区的延迟**，使得共享内存像每个线程块的高速缓存或者中间结果暂存器那样高效。

如果想要实现线程之间通信，那么还需要一种机制来实现线程之间的同步。例如，如果线程A将一个值写入到共享内存，并且我们希望线程B对这个值进行一些操作，那么只有当线程A的写入操作完成之后，线程B才能开始执行它的操作。如果没有同步，那么将发生竞态条件(race condition)。

下面将通过一个矢量的点积运算来详细介绍共享内存和同步。矢量点积运算为矢量相乘结束后将值相加起来以得到一个标量输出值。例如对两个包含4个元素的矢量进行点积运算:
$$
(x_1, x_2, x_3, x_4) * (y_1, y_2, y_3, y_4) = x_1y_1 + x_2y_2 + x_3y_3 + x_4y_4
$$
由于最终结果是所有乘积的总和，因此每个线程要保存它所计算的乘积的加和。下面代码实现了点积函数的第一个步骤:

```c
#include "../common/book.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    //设置cache中相应位置上的值
    cache[cacheIndex] = temp;
}
```

代码中声明了一个共享内存缓冲区，名字为cache。这个缓冲区将保存每个线程计算的加和值。我们将cache数组的大小声明为threadsPerBlock，这样线程块中每个线程都能将它计算的临时结果保存到某个位置上。之前在分配全局内存时，我们为每个执行核函数的线程都分配了足够的内存，即线程块的数量乘以threadsPerBlock。但对于共享变量来说，由于编译器将为每个线程块生成共享变量的一个副本，因此只需根据线程块中线程的数量来分配内存。

我们需要将cache中所有的值相加起来。在执行这个运算时，需要通过一个线程来读取保存在cache中的值。由于race condition，我们需要使用下面的代码来确保对所有共享数组cache[]的写入操作在读组cache之前完成:

```c
//对线程块中的线程进行同步
__syncthreads();
```

这个函数调用将确保线程块中的每个线程都执行完__syncthreads()前面的语句后，才会执行下一条语句。

这时，我们可以将其中的值相加起来(称为归约Reduction)。代码的基本思想是每个线程将cache[]中的两个值相加起来，然后将结果保存回cache[]。由于每个线程都将两个值合并为一个值，那么在完成这个步骤后，得到的结果数量就是计算开始时数值数量的一半。下一个步骤中我们对这一半数值执行相同的操作，在将这种操作执行log2(threadsPerBlock)步骤后，就能得到cache[]中所有值的总和。对于这个示例来说，我们在每个线程块中使用了256个线程，因此需要8次迭代将cache[]中的256个值归约为1个值。这个归约过程的实现可以表示为以下代码:

```c
//对于归约运算来说，以下代码要求threadsPerBlock必须时2的指数
int i = blockDim.x / 2;
while(i != 0){
    if(cacheIndex < i){
        cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
}
```

再结束了while()循环后，每个线程块都得到了一个值，这个值位于cache[]的第一个元素中，并且就等于该线程块中两两元素乘积的加和。然后，我们将这个值保存到全局内存并结束核函数:

```c
if(cacheIndex == 0){
    c[blockIdx.x] = cache[0];
}
```

只让cacheIndex为0的线程执行保存操作时因为每个线程块只有一个值写入到全局内存，因此每个线程块只需要一个线程来执行这个操作。最后，由于每个线程块都只写入一个值到全局数据c[]中，因此可以通过blockIdx来索引这个值。

点积运算的最后一个步骤就是计算c[]中所有元素的总和。像GPU这种大规模的并行机器在执行最后的归约步骤时，通常会浪费计算资源，因为此时的数据集往往会非常小。因此，我们可以将执行控制返回给主机，并且由CPU来完成最后一个加法步骤。

下面给出了完整的代码实现:

```c
#include "../common/book.h"

#define imin(a, b) (a < b? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0;
    while(tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    //设置cache中相应位置上的值
    cache[cacheIndex] = temp;
    
    //对线程块中的线程进行同步
    __syncthreads();
    
    //对于归约来说，以下代码要求threadsPerBlock必须是2的指数
    int i = blockDim.x / 2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads(); //循环中更新了变量cache，所以需要在下一次循环前进行同步。该同步语句需要所有的线程都必须运行才行。如果有线程不能运行这一处代码，会导致其他线程永远等待。
        i /= 2;
    }
    if(cacheIndex == 0){
        c[blockIndex.x] = cache[0];
    }

}

int main(){
    float *a, *b, c, *partial_c;
    float *dev_c, *dev_b, *dev_partial_c;
    
    //在CPU上分配内存
    a = (float*) malloc(N*sizeof(float));
    b = (float*) malloc(N*sizeof(float));
    partial_c = (float*) malloc(blocksPerGrid * sizeof(float));
    
    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));
    
    //填充主机内存
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }
    
    //将数组"a"和"b"复制到GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    
    //将数组"c"从GPU复制到CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    //在CPU上完成最终的求和运算
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }
    
    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g? \n", c, 2 * sum_square((float) (N - 1)));
    
    //释放GPU上的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    
    //释放CPU上的内存
    free(a);
    free(b);
    free(partial_c);
}
```

