---
title: 		 	"CUDA学习(五)"
subtitle:    	""
description: 	""
date:        	2024-02-23
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

本章将学习如何分配和使用纹理内存(texture memory)。和常量内存一样，纹理内存是另一种类型的只读内存，在特定的访问模式中，纹理内存同样能够提升性能并减少内存流量。虽然纹理内存最初是针对传统的图形处理应用程序而设计的，但在某些GPU计算应用程序中同样非常有用。

<!-- more -->

与常量内存相似的是，纹理内存同样缓存在芯片上，因此在某些情况中，它能够减少对内存请求并提供更高效的内存带宽。纹理缓存是专门为那些在内存访问模式中存在大量空间局部性(spatial locality)的图形应用程序而设计的。在某个计算应用程序中，这意味着一个线程读取的位置可能与邻近线程读取的位置“非常接近”。



## 热传导模拟

本章用一个简单的热传导模拟的模型来介绍如何使用纹理内存。



### 简单的传热模型

我们构造一个简单的二维热传导模拟。首先假设有一个矩形房间，将其分成一个格网，每个格网中随机散步一些“热源”，他们有着不同的温度。

![CUDA_5_1](CUDA/CUDA_5_1.png)

在给定了矩形格网以及热源分布后，我们可以计算格网中每个单元格的温度随时间的变化情况。为了简单，热源单元本身的温度将保持不变。在时间递进的每个步骤中，我们假设热量在某个单元及其邻接单元之间“流动”。如果某个单元的临界单元的温度更高，那么热量将从邻接单元传导到该单元，相反地，如果某个单元的温度比邻接单元的温度高，那么它将变冷。

在热传导模型中，我们对单元中新温度的计算方法为，将单元与邻接单元的温差相加起来，然后加上原有温度。
$$
T_{NEW} = T_{OLD} + \sum_{NEIGHBOURS}k(T_{NEIGHBORS} - T_{OLD})
$$
在上面的计算单元温度的等式中，常量k表示模拟过程中热量的流动速率，k值越大，表示系统会更快地达到稳定温度，而k值越小，则温度梯度将存在更长时间。由于我们只考虑4个邻接单元(上、下、左、右)并且等式中的k和$$T_{OLD}$$都是常数，因此把上述公式展开表示为:
$$
T_{NEW} = T_{OLD} + k(T_{TOP} + T_{BOTTOM} + T_{LEFT} + T_{RIGHT} - 4T_{OLD})
$$


### 温度更新的计算

以下是更新流程的基本介绍:

1. 给定一个包含初始输入温度的格网，将其中作为热源的单元温度值复制到格网相应的单元中来覆盖这些单元之前计算出来的温度，确保“加热单元将保持恒温”的条件。这个复制操作在copy_const_kernel()中执行。
2. 给定一个输入温度格网，根据新的公式计算出输出温度格网。这个更新操作在blend_kernel()中执行。
3. 将输入温度格网和输出温度格网交换，为下一个步骤的计算做好准备。当模拟下一个时间步时，在步骤2中计算得到的输出温度格网将成为步骤1中的输入温度格网。

下面是两个函数的具体实现:

```c
__global__ void copy_const_kernel(float *iptr, const float *cptr){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    if(cptr[offset] != 0){
        iptr[offset] = cptr[offset];
    }
}

//为了执行更新操作，可以在模拟过程中让每个线程都负责计算一个单元。
//每个线程都将读取对应单元及其邻接单元的温度值，执行更新运算，然后计算得到新值来更新温度。
__global__ void blend_kernel(float *outSrc, const float *inSrc){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    int left = offset - 1;
    int right = offset + 1;
    if(x == 0) left++;
    if(x == DIM - 1) right--;
    
    int top = offset - DIM;
    int bottom = offset + DIM;
    if(y == 0) bottom += DIM;
    if(y == DIM - 1) top -= DIM;
    
    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}
```



### 模拟过程动态演示

剩下的代码主要是设置好单元，然后显示热量的动画输出

```c
#include "cuda.h"
#include "../common/book.h"
#include "cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

//更新函数中需要的全局变量
struct DataBlock{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;
    
    cudaEvent_t start, stop;
    float totalTime;
    float frames;
}

void anim_gpu(DataBlock *d, int ticks){
    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;
    
    for(int i = 0; i < 90; 0++){
        copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        blend_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
        swap(d->dev_inSrc, d->dev_outSrc);
    }
    float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
    
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Averaged Time per frame: $3.1f ms \n", d->totalTime / d->frames);
    
}

void anim_exit(DataBlock *d){
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);
    
    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));
    
    //假设float类型的大小为4个字符(即rgba)
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));
    
    float *temp = (float*)malloc(bitmap.image_size());
    for(int i = 0; i < DIM*DIM; i++){
        temp[i] = 0;
        int x = i % DIM;
        int y =i / DIM;
        if((x > 300) && (x < 600) && (y > 310) && (y < 601)){
            temp[i] = MAX_TEMP;
        }
        temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
        temp[DIM * 700 + 100] = MIN_TEMP;
        temp[DIM * 300 + 300] = MIN_TEMP;
        temp[DIM * 200 + 700] = MIN_TEMP;
        for(int y = 800; y < 900; y++){
            for(int x = 400; x < 500; x++){
                temp[x * y * DIM] = MIN_TEMP;
            }
        }
        HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
        
        free(temp);
        
        bitmap.anim_and_exit((void (*) (void*, int)) anim_gpu, (void (*) (void*)) anim_exit);
    }
    
}
```



### 使用纹理内存

如果要使用纹理内存，首先要将输入的数据声明为texture类型的引用。我们使用浮点类型纹理的引用，因为温度数值是浮点类型。

```c
//这些变量将位于GPU上
texture<float> texConstSrc;
texture<float> textIn;
texture<float> textOut;
```

下一个需要注意的问题是，在为这三个缓冲区分配了GPU内存后，需要通过cudaBindTexture()将这些变量绑定到内存缓冲区。这相当于告诉CUDA运行时两件事情：

- 我们希望将制定的缓冲区作为纹理来使用。
- 我们希望将纹理引用作为纹理的“名字”。

在热传导模拟中分配了这三个内存后，需要将这三个内存绑定到之前声明的纹理引用(texConstSrc, textIn, textOut)。

```c
HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));
HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));
HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));

HANDLE_ERROR(cudaBindTexture(NULL, textConstSrc, data.dev_constSrc, imageSize));
HANDLE_ERROR(cudaBindTexture(NULL, textIn, data.dev_inSrc, imageSize));
HANDLE_ERROR(cudaBindTexture(NULL, textOut, data.dev_outStc, imageSize));
```

此时，纹理变量已经设置好，可以启动核函数。然而，当读取核函数中的纹理时，需要通过特殊的函数来告诉GPU将读取请求转发到纹理内存而不是标准的全局内存。因此，当读取内存时不再使用方括号从缓冲区读取，而是将blend_kernel()函数内修改为使用tex1Dfetch()。

tex1Dfetch()实际上是一个编译器内置函数。由于纹理引用必须声明为文件作用域内的全局变量，因此我们不再将输入缓冲区和输出缓冲区作为参数传递给blend_kernel()，因为编译器需要在编译时知道text1Dfetch()应该对哪些纹理采样。我们需要将一个布尔标志dstOut传递给blend_kernel()，这个标志会告诉我们使用那个缓冲区作为输入，以及哪个缓冲区作为输出。以下是对blend_kernel()的修改。

```c
__global__ void blend_kernel(float *dst, bool dstOut){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    int left = offset - 1;
    int right = offset + 1;
    if(x == 0) left++;
    if(x == DIM - 1) right--;
    
    int top = offset - DIM;
    int bottom = offset + DIM;
    if(y == 0) bottom += DIM;
    if(y == DIM - 1) top -= DIM;
    
    //替换该行代码
    // outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
    float t, l, c, r, b;
    if(dstOut){
        t = text1Dfetch(textIn, top);
        l = text1Dfetch(textIn, left);
        c = text1Dfetch(textIn, offset);
        r = text1Dfetch(textIn, right);
        b = text1Dfetch(textIn, bottom);
    }
    else{
        t = text1Dfetch(textOut, top);
        l = text1Dfetch(textOut, left);
        c = text1Dfetch(textOut, offset);
        r = text1Dfetch(textOut, right);
        b = text1Dfetch(textOut, bottom);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}
```

由于核函数copy_const_kernel()将读取包含热源位置和温度的缓冲区，因此同样需要修改为从纹理内存而不是从全局内存中读取：

```c
__global__ void copy_const_kernel(float *iptr){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    float c = text1Dfetch(textConstSrc, offset);
    if(c != 0){
        iptr[offset] = c;
    }
}
```

由于blend_kernel()的函数原型被修改为接收一个标志，并且这个标志表示在输入缓冲区与输出缓冲区之间的切换，因此需要对anim_gpu()函数进行相应的修改。现在，不是交换缓冲区，而是在每组调用之后通过设置dstOut = !dstOut来进行切换：

```c
void anim_gpu(DataBlock *d, int ticks){
    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap *bitmap = d->bitmap;
    
    //for(int i = 0; i < 90; 0++){
    //    copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
    //    blend_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
    //    swap(d->dev_inSrc, d->dev_outSrc);
    //}
    //float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
    
    //由于tex是全局并且有界的，因此我们必须通过一个标志来选择每次迭代中哪个是输入/输出
    volatile bool dstOut = true;
    for(int i = 0; i < 90; i++){
        float *in, *out;
        if(dstOut){
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else{
            out = d->dev_inSrc;
            in = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks, threads>>>(in);
        blend_kernel<<<blocks, threads>>>(out, dstOut);
        dstOut = !dstOut;
    }
    
    
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Averaged Time per frame: $3.1f ms \n", d->totalTime / d->frames);
}
```

对热传导函数的最后一个修改就是在应用程序运行结束后的清理工作。不仅要释放全局缓冲区，还需要清楚与纹理的绑定：

```c
void anim_exit(DataBlock *d){
    cudaUnbindTexture(textIn);
    cudaUnbindTexture(textOut);
    cudaUnbindTexture(texConstSrc);
    
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);
    cudaFree(d->dev_constSrc);
    
    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}
```



### 使用二维纹理内存

在多数情况下，二维内存空间是非常有用的。首先，要修改纹理引用的声明。默认的纹理引用都是一维的，因此我们需要增加代表维数的参数2，这表示声明的是一个二维纹理引用。

```c
texture<float, 2> texConstSrc;
textture<float, 2> texIn;
textture<float, 2> texOut;
```

二维纹理将简化blend_kernel()方法的实现。虽然我们需要将tex1Dfeth()调用修改为text2D()调用，但却不再需要通过线性化offset变量以计算top、left、right和bottom等偏移。当使用二维纹理时，可以直接通过x和y来访问纹理。而且当使用tex2D()时，我们不再需要担心发生溢出问题。如果x或y小于0，那么tex2D()将返回0处的值。同理，如果某个值大于宽度，那么tex2D()将返回位于宽度处的值。这些建华带来的好处之一就是核函数的代码将变得更加简单。

```c
__global__ void blend_kernel(float *dst, bool dstOut){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float t, l, c, r, b;
    if(dstOut){
        t = text2D(textIn, x, y - 1);
        l = text2D(textIn, x - 1, y);
        c = text2D(textIn, x, y);
        r = text2D(textIn, x + 1, y);
        b = text2D(textIn, x, y + 1);
    }
    else{
        t = text2D(textOut, x, y - 1);
        l = text2D(textOut, x - 1, y);
        c = text2D(textOut, x, y);
        r = text2D(textOut, x + 1, y);
        b = text2D(textOut, x, y + 1);
    }
    dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}
```

我们也需要对copy_const_kernel()中进行相应的修改。与核函数blend_kernel()类似的是，我们不再需要通过offset来访问纹理，而是只需使用x和y来访问热源的常量。

```c
__global__ void copy_const_kernel(float *iptr){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    float c = text2D(textConstSrc, x, y);
    if(c != 0){
        iptr[offset] = c;
    }
}
```

在main()需要对纹理绑定调用进行修改，并告诉运行时：缓冲区将被视为二维纹理而不是一维纹理：

```c
int main(){
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));
    
    //假设float类型的大小为4个字符(即rgba)
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));
    
    
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float * DIM)));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texin, data.dev_inSrc, desc, DIM, DIM, sizeof(float * DIM)));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float * DIM)));
    
    
    
    float *temp = (float*)malloc(bitmap.image_size());
    for(int i = 0; i < DIM*DIM; i++){
        temp[i] = 0;
        int x = i % DIM;
        int y =i / DIM;
        if((x > 300) && (x < 600) && (y > 310) && (y < 601)){
            temp[i] = MAX_TEMP;
        }
        temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
        temp[DIM * 700 + 100] = MIN_TEMP;
        temp[DIM * 300 + 300] = MIN_TEMP;
        temp[DIM * 200 + 700] = MIN_TEMP;
        for(int y = 800; y < 900; y++){
            for(int x = 400; x < 500; x++){
                temp[x * y * DIM] = MIN_TEMP;
            }
        }
        HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
        
        free(temp);
        
        bitmap.anim_and_exit((void (*) (void*, int)) anim_gpu, (void (*) (void*)) anim_exit);
    }
    
}
```

虽然我们需要通过不同的函数来告诉运行时绑定一维纹理还是二维纹理，但是可以通过同一个函数cudaUnbindTexture()来取消纹理绑定。所以执行释放操作的函数anim_exit()可以保持不变。
