---
title:       	"CUDA学习(八)"
subtitle:    	""
description: 	""
date:        	2024-02-29
#type:		 	docs
featured: 	 	false
draft: 		 	true
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

本章将介绍如何分配和使用零拷贝内存(Zero-Copy Memory)，如何在同一个应用程序中使用多个GPU，以及如何分配和使用可移动的固定内存(Portable Pinned Memory)。

<!-- more -->

## 零拷贝主机内存

上一章介绍了固定内存（页锁定内存），这种新型的主机内存能够确保不会交换出物理内存。我们通过cudaHostAlloc()来分配这种内存，并且传递参数cudaHostAllocDefault来获得默认的固定内存。本章会介绍在分配固定内存时可以使用其他参数值。除了cudaHostAllocDefault外，还可以传递的标志之一是cudaHostAllocMapped。通过cudaHostAllocMapped分配的主机内存也是固定的，它与通过cudaHostAllocDefault分配的固定内存有着相同的属性，特别是当它不能从物理内存中交换出去或者重新定位时。但这种内存除了可以用于主机与GPU之间的内存复制外，还可以打破主机内存规则之一：可以在CUDA C核函数中直接访问这种类型的主机内存。由于这种内存不需要复制到GPU，因此也被称为零拷贝内存。



### 通过零拷贝内存实现点积运算

通常，GPU只能访问GPU内存，而CPU也只能访问主机内存。但在某些环境中，打破这种规则或许能带来更好的效果。下面仍然给出一个矢量点积运算来进行介绍。这个版本不将输入矢量显式复制到GPU，而是使用零拷贝内存从GPU中直接访问数据。我们将编写两个函数，其中一个函数是对标准主机内存的测试，另一个函数将在GPU上执行归约运算，并使用零拷贝内存作为输入缓冲区和输出缓冲区。首先是点积运算的主机内存版本:

```c
float malloc_test(int size){
    //首先创建计时事件，然后分配输入缓冲区和输出缓冲区，并用数据填充输入缓冲区。
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;
    
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    
    //在CPU上分配内存
    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
    
    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));
    
    //用数据填充主机内存
    for(int i = 0; i < size; i++){
        a[i] = i;
        b[i] = i * 2;
    }
    
    //启动计时器，将输入数据复制到GPU，执行点积核函数，并将中间计算结果复制回主机。
    HANDLE_ERROR(cudaEventRecord(start, 0));
    //将数组“a“和”b”复制到GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    //将数组“c”从GPU复制到CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    //停止计时器
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    //将中间计算结果相加起来，并释放输入缓冲区和输出缓冲区
    //结束CPU上的计算
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));
    
    //释放CPU上的内存
    free(a);
    free(b);
    free(partial_c);
    
    //释放事件
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    
    printf("Value calculated: %f\n", c);
    
    return elapsedTime;
}
```

使用零拷贝内存的版本是非常类似的多，只是在内存分配上有所不同：

```c
float cuda_host_alloc_test(int size){
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;
    
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    
    //在CPU上分配内存
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocMapped));
    
    //用数据填充主机内存
    for(int i = 0; i < size; i++){
        a[i] = i;
        b[i] = i * 2;
    }

```

使用cudaHostAlloc()时，通过参数flags来指定内存的其他行为。cudaHostAllocMapped这个标志告诉运行时将从GPU中访问这块内存。这个标志意味着分配零拷贝内存。对于这两个输入缓冲区，我们还制定了标志cudaHostAllocWriteCombined。这个标志运行时应该将内存分配为“合并式写入（Write-Combined）”内存。这个标志并不会改变应用程序的功能，但却可以显著地提升GPU读取内存时的性能。然而，当CPU也要读取这块内存时，“合并式写入”会显得低效，因此在决定是否使用这个标志之前，必须首先考虑应用程序的可能访问模式。

在使用标志cudaHostAllocMapped来分配主机内存后，就可以从GPU中访问这块内存。然而，GPU的虚拟内存空间与CPU是不同的，因此在**GPU上访问它们与在CPU上访问它们有着不同的地址。调用cudaHostAlloc()将返回这块内存在CPU上的指针，因此需要调用cudaHostGetDevicePointer()来获得这块内存在GPU上的有效指针。**这些指针将被传递给核函数，并在随后由GPU对这块内存执行读取和写入等操作。即使dev_a、dev_b和dev_partial_c都位于主机上，但对于核函数来说，它们看起来就像GPU内存一样，这正是由于调用了cudaHostGetDevicePointer()。由于部分计算结果已经位于主机上，**因此就不再需要通过cudaMemcpy()将它们从设备上复制回来。**

```c
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

    //启动计时器以及核函数
    HANDLE_ERROR(cudaEventRecord(start, 0));
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
    //不再需要通过cudaMemcpy()将它们从设备上复制回来
    HANDLE_ERROR(cudaThreadSynchronize());

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    //结束GPU上的操作
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

    //在使用cudaHostAlloc()的点积运算代码中，唯一剩下的事情就是执行释放操作
    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(partial_c));

    //释放事件
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    printf("Value calculated: %f\n", c);

    return elapsedTime;
}
```

无论cudaHostAlloc()中使用什么标志，总是按照相同的方式来释放内存，即只需调用cudaFreeHost()。剩下的工作就是观察main()如何将这些代码片段组合在一起。

```c
int main(){
    cudaDeviceProp prop;
    int which Device;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    if(prop.canMapHostMemory != 1){
        printf("Device, cannot map memory. \n");
    }
    
    //如果设备支持零拷贝内存，那么接下来就是将运行时置入能分配零拷贝内存的状态
    //通过调用cudaSetDeviceFlags()来实现这个操作，并且传递标志值cudaDeviceMapHost来表示我们希望设备映射主机内存
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    
    //运行两个测试，分别显示二者的执行时间，并推出应用程序：
    float elapsedTime = malloc_test(N);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
    elapsedTime = cuda_host_alloc_test(N);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
}
```

下面是给出的核函数

```c
#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while(tid < size){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    //设置cache中的值
    cache[cacheIndex] = temp;
    
    //同步这个线程块中的线程
    __syncthreads();
    
    //对于归约运算， threadsPerBlock必须为2的幂
    int i = blockDim.x / 2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if(cacheIndex == 0){
        c[blockIdx.x] = cache[0];
    }
}
```



## 使用多个GPU

我们将把点积应用程序修改为使用多个GPU。为了降低编码难度，我们将在上一个结构中把计算点积所需的全部数据都相加起来。

```c
struct DataStruct{
    int deviceID;
    int size;
    float *a;
    float *b;
    float returnValue;
}
```

这个结构包含了在计算点积时使用的设备标识，以及输入缓冲区的大小和指向两个输入缓冲区的指针a和b。最后，它还包含了一个成员用于保存a和b的点积运算结果。

要使用N个GPU，我们首先需要准确地知道N值是多少。因此，在应用程序的开头调用cudaDeviceCount()，从而判断在系统中安装了多少个支持CUDA的处理器。

```c
int main(){
    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    if(deviceCount < 2){
        printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    }
    
    //为输入缓冲区分配标准的主机内存，并按照之前的方式填充
    float *a = (float*)malloc(sizeof(float) * N);
    HANDLE_NULL(a);
    float *b = (float*)malloc(sizeof(float) * N);
    HANDLE_NULL(b);
    
    //用数据填充主机内存
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }
```

通过CUDA运行API来使用多个GPU时，要意识到每个GPU都需要由一个不同的CPU线程来控制。由于之前只是用了单个GPU，因此不需要担心这个问题。我们将多线程代码的大部分复杂性都移入到辅助代码文件book.h中。在精简了代码后，我们需要做的就是填充一个结构来执行计算。虽然在系统中可以有任意数量的GPU，但为了简单，在这里只使用两个：

```c
    DataStruct data[2];

    data[0].deviceID = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].size = N / 2;
    data[1].a = a + N / 2;
    data[1].b = b + N / 2;
```

我们将其中一个DataStruct变量传递给辅助函数start_thread()。此外，还将一个函数指针传给了start_thread()，新创建的线程将调用这个函数，这个示例中的线程函数为routine()。函数start_thread()将创建一个新的线程，这个线程将调用routine()，并将DataStruct变量作为参数传递进去。在应用程序的默认线程中也将调用routine()(因此只多创建了一个线程)。

```c
    CUTThread thread = start_thread(routine, &(data[0]));
    routine(&(data[1]));

    //通过调用end_thread()，主应用程序线程将等待另一个线程执行完成。
    end_thread(thread);

    //由于这两个线程都在main()的这个位置上执行完成，因此可以安全地释放内存并显示结果。
    free(a);
    free(b);
	
	//我们要将每个线程的计算结果相加起来。
    printf("Value calculated: %f\n", data[0].returnValue + data[1].returnValue);
}
```

在声明routine()时指定该函数带有一个void\*参数，并返回void\*，这样在start_thread()部分代码保持不变的情况下可以任意实现线程函数。

```c
void* routine(void *pvoidData){
    DataStruct *data = (DataStruct*)pvoidData;
    HANDLE_ERROR(cudaSetDevice(data->deviceID));

```

除了调用cudaSetDevice()来指定希望使用的CUDA设备外，routine()的实现非常类似于之前提到的malloc_test()。我们为输入数据和临时计算结果分别分配了内存，随后调用cudaMemcpy()将每个输入数组复制到GPU。

```c
    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //在CPU上分配内存
    a = data->a;
    b = data->b;
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, size * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    //将数组“a”和“b”复制到GPU上
    HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

    //启动点积核函数，复制回计算结果，并且结束CPU上的操作
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    //将数组“c”从GPU复制回CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    //结束CPU上的操作
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    //释放CPU侧的内存
    free(partial_c);

    data->returnValue = c;
}
```



## 可移动的固定内存

我们可以将固定内存分配为可移动的，这意味着可以在主机线程之间移动这块内存，并且每个线程都将其视为固定内存。要达到这个目的，需要使用cudaHostAlloc()来分配内存，并且在调用时使用一个新的标志：cudaHostAllocPortable。这个标志可以与其他标志一起使用，例如cudaHostAllocWriteCombined和cudaHostAllocMapped。这意味着在分配主机内存时，可将其作为可移动、零拷贝以及合并式写入等的任意组合。

为了说明可移动固定内存的作用，我们将进一步修改使用多GPU的点积运算应用程序。

```c
int main(){
    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    if(deviceCount < 2){
        printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    }
    
    cudaDeviceProp prop;
    for(int i = 0; i < 2; i++){
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        if(prop.canMapHostMemory != 1){
            printf("Devide %d cannot map memory.\n", i);
        }
    }
    
    float *a, *b;
    HANDLE_ERROR(cudaSetDevice(0));
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    HANDLE_ERROR(cudaHostAlloc((void**)&a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));
```

在使用cudaHostAlloc()分配页锁定内存时，首先要通过调用cudaSetDevice()来初始化设备。我们将新介绍的标志cudaHostAllocPortable传递给这两个内存分配操作。由于这些内存是在调用了cudaSetDevice(0)之后才分配的，因此，如果没有将这些内存指定为可移动的内存，那么只有第0个CUDA设备会把这些内存视为固定内存。

继续之前的应用程序，为输入矢量生成数据，并采用之前的示例的方式来准备DataStruct结构。

```c
    //用数据填充主机内存
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    //为使用多线程做好准备
    DataStruct data[2];
    data[0].deviceID = 0;
    data[0].offset = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[1].deviceID = 1;
    data[1].offset = N / 2;
    data[1].size = N / 2;
    data[1].a = a;
    data[1].b = b;

    //创建第二个线程，并调用routine()开始在每个设备上执行计算
    CUTThread thread = start_thread(routine, &(data[1]));
    routine(&(data[0]));
    end_thread(thread);

    //由于主机内存时由CUDA运行时分配的，因此需要用cudaFreeHost()而不是free()来释放它
    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));

    printf("Value calculated: %f\n", data[0].returnValue + data[1].returnValue);
}
```

为了在多GPU应用程序中支持可移动的固定内存和零拷贝内存，我们需要对routine()的代码进行两处修改。

```c
void* routine(void *pvoidData){
    DataStruct *data = (DataStruct*)pvoidData;
    if(data->deviceID != 0){
        HANDLE_ERROR(cudaSetDevice(data->deviceID));
        HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
```

在多GPU版本的代码中，我们需要在routine()中调用cudaSetDevice()，从而确保每个线程控制一个不同的GPU。另一方面，在这个示例中，我们已经在主线程中调用了一次cudaSetDevice()。这么做的原因时为了在main()中分配固定内存。因此，我们只希望在还没有调用cudaSetDevice()的设备上调用cudaSetDevice()和cudaSetDeviceFlags()。也就是，如果devideID不是0，那么将调用这两个函数。虽然在第0个设备上再次调用这些函数会使代码更简洁，但是这种做法是错误的。一旦在某个线程上设置了这些设备，那么将不能再次调用cudaSetDevice()，即便传递的是相同的设备标识符。

除了使用可移动的固定内存外，我们还使用了零拷贝内存，一边从GPU中直接访问这些内存。因此，我们使用cudaHostGetDevicePointer()来获得主机内存的有效设备指针，这与前面零拷贝示例中采用的方法一样。然而，你可能会注意到使用了标准的GPU内存来保存临时计算结果。这块内存同样是通过cudaMalloc()来分配的。

```c
    int size = data->size;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //在CPU上分配内存
    a = data->a;
    b = data->b;
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    //计算GPU读取数据的偏移量“a”和“b”
    dev_a += data->offset;
    dev_b += data->offset;
    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    //将数组“c“从GPU复制回CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    //结束在CPU上的操作
    c = 0;
    for(int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }
    HANDLE_ERROR(cudaFree(dev_partial_c));

    //释放CPU上的内存
    free(partial_c);

    data->returnValue = c;
}
```

