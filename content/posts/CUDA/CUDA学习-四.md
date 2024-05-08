---
title:       	"CUDA学习(四)"
subtitle:    	""
description: 	""
date:        	2024-02-18
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

这一章会介绍如何在CUDA C中使用常量内存、了解常量内存的性能特性以及学习如何使用CUDA事件来测量应用程序的性能。

<!-- more -->



## 常量内存

到目前为止，我们知道CUDA C程序中可以使用全局内存和共享内存。但是，CUDA C还支持另一种类型的内存，即常量内存。常量内存用于保存在核函数执行期间不会发生变化的数据。在某些情况下，用常量内存来替换全局内存能有效地减少内存带宽。



### 在GPU上实现光线跟踪

我们给出一个简单的光线跟踪应用程序示例来学习。由于OpenGL和DirectX等API都不是专门为了实现光线跟踪而设计的，因此我们必须使用CUDA C来实现基本的光线跟踪器。本示例构造的光线跟踪器非常简单，旨在学习常量内存的使用上(并不能通过示例代码来构建一个功能完备的渲染器)。这个光线跟踪器只支持一组包含球状物体的场景，并且相机被固定在了Z轴，面向原点。此外，示例代码也不支持场景中的任何照明，从而避免二次光线带来的复杂性。代码也不计算照明效果，而只是为每个球面分配一个颜色值，如果它们是可见的，则通过某个预先计算的值对其着色。

光线跟踪器将从每个像素发射一道光线，并且跟踪到这些光线会命中哪些球面。此外，它还将跟踪每道命中光线的深度。当一道光线穿过多个球面时，只有最接近相机的球面才会被看到。这个代码的光线跟踪器会把相机看不到的球面隐藏起来。

通过一个数据结构对球面建模，在数据结构中包含了球面的中心坐标(x, y, z)，半径radius，以及颜色值(r, g, b)。

```c
#define INF 2e10f

struct sphere{
    float r, g, b;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n){
        float dx = ox - x;
        float dy = oy - y;
        if(dx * dx + dy * dy < radius * radius){
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
}
```

这个结构中定义了一个方法hit(float ox, float oy, float *n)。对于来自(ox, oy)处像素的光线，这个方法将计算光线是否与这个球面相交。如果光线与球面相交，那么这个方法将计算从相机到光线命中球面处的距离。我们需要这个信息，因为当光线命中多个球面时，只有最接近相机的球面才会被看见。

main()函数遵循了与前面示例大致相同的代码结构。

```c
#include "cuda.h"
#include "../common/book.h"
#include "cpu_bitmap.h"

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20

Sphere *s;
int main(){
    //记录起始时间
    cudaEvent_ start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    
    //在GPU上分配内存以计算输出位图
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    
    //为Sphere数据集分配内存
    HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));
}
```

在分配输入数据和输出数据的内存后，我们将随机地生成球面的中心坐标，颜色以及半径。

```c
//分配临时内存，对其初始化，并复制到GPU上的内存，然后释放临时内存
Sphere *temp_s = (Sphere*) malloc(sizeof(Sphere) * SPHERES);
for(int i = 0; i < SPHERES; i++){
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500;
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(100.0f) + 20;
}
```

当前，程序将生成一个包含20个球面的随机数组，但这个数量值是通过一个#define宏指定的，因此可以相应的做出调整。

通过cudaMemcpy()将这个球面数组复制到GPu，然后释放临时缓冲区。

```c
HANDLE_ERROR(cudaMemcpy(s, temps, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
free(temp_s);
```

现在，输入数据位于GPU上，并且我们已经为输出数据分配好了空间，因此可以启动核函数。

```c
//从球面数据汇总生成一张位图
dim3 grids(DIM / 16, DIM / 16);
dim3 threads(16, 16);
kernel<<<grids, threads>>>(dev_bitmap);
```

这个核函数将执行光线跟踪计算并从输入的一组球面中为每个像素计算颜色数据。最后，我们把输出图像从GPU中复制回来，并显示它。我们还要释放所有已经分配但还未释放的内存。

```c
//将位图从GPU复制回到CPU以显示
HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
bitmap.display_and_exit();

//释放内存
cudaFree(dev_bitmap);
cudaFree(s);
```

下面的代码将介绍如何实现核函数的光线跟踪算法。每个线程都会为输出影像中的一个像素计算颜色值，因此我们遵循一种惯用的方式，计算每个线程对应的x坐标和y坐标，并且根据这两个坐标来计算输出缓冲区的偏移。此外，我们还将把图像坐标(x, y, z)偏移DIM/2，这样z轴将穿过图像的中心。

```c
__global__ void kernel(unsigned char *ptr){
    //将threadIdx/BlockIdx映射到像素位置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = x - DIM / 2;
    float oy = y - DIM / 2;
    
    //由于每条光线都需要判断与球面相交的情况，因此我们现在对球面数组进行迭代，并判断每个球面的命中情况。
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for(int i = 0; i < SPHERES; i++){
        float n;
        float t = s[i].hit(ox, oy, &n);
        if(t > maxz){
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }
    //在判断了每个球面的相交情况后，可以将当前颜色值保存到输出图像中
    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 1] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}
```



#### 通过常量内存来实现光线跟踪

上述代码中并没有提到常量内存。下面的代码将使用常量内存来修改这个例子。由于常量内存无法修改，因此显然无法用常量内存来保存输出图像的数据。在这个示例中只有一个输入数据，即球面数组，因此应该将这个数据保存到常量内存中。

声明数组时，要在前面加上\_\_constant\_\_修饰符。

```c
__constant__ Sphere s[SPHERES];
```

最初的示例中，我们声明了一个指针，然后通过cudaMalloc()来为指针分配GPU内存。当我们将其修改为常量内存时，同样要将这个声明修改为在常量内存中静态地分配空间。我们不再需要对球面数组调用cudaMalloc()或cudaFree()。而是在编译时为这个数组提交一个固定的大小。将main()函数修改为常量内存的代码如下:

```c
int main(){
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    
    //在GPU上分配内存以计算输出位图
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    
    //分配临时内存，对其初始化，并复制到GPU上的内存，然后释放临时内存
    Sphere *temp_s = (Sphere*) malloc(sizeof(Sphere) * SPHERES);
	for(int i = 0; i < SPHERES; i++){
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
	}
    
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    
    free(temp_s);
    
    //从球面数据中生成一张位图
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);
    
    //将位图从GPU复制回到CPU以显示
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    bitmap.display_and_exit();

    //释放内存
    cudaFree(dev_bitmap);
}
```

cudaMemcpyToSymbol()与参数为cudaMemcpyHostToDevice()的cudaMemcpy()之间唯一差异时cudaMemcpyToSymbol()会复制到常量内存，而cudaMemcpy()会复制到全局内存。



### 使用事件来测量性能

如何判断常量内存对程序性能有着多大影响？最简单的方式就是判断哪个版本的执行事件更短。使用CPU计时器或者操作系统中的某个计时器会带来各种延迟。为了测量GPU在某个任务上话费的时间，我们将使用CUDA的事件API。

CUDA中的事件本质上是一个GPU时间戳，这个时间戳是在用户指定的时间点上记录的。由于GPU本身支持记录时间戳，因此就避免了当使用CPU定时器来统计GPU执行的事件时可能遇到的诸多问题。比如，下面的代码开头告诉CUDA运行时记录当前的时间，首先创建一个事件，然后记录这个事件。

```c
cudaEvent_t start;
cudaEventCreate(&start);
cudaEventRecord(start, 0);
```

要统计一段代码的执行时间，不仅要创建一个起始事件，还要创建一个结束事件。当在GPU上执行某个工作时，我们不仅要告诉CUDA运行时记录起始时间，还要记录结束时间:

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

//在GPU执行一些工作

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop); //表示stop事件之前的所有GPU工作已经完成，可以安全读取在stop中保存的时间戳
```

下面是对光线跟踪器进行性能测试的代码:

```c
int main(){
    
    //记录起始时间
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
    
    
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    
    //在GPU上分配内存以计算输出位图
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    
    //分配临时内存，对其初始化，并复制到GPU上的内存，然后释放临时内存
    Sphere *temp_s = (Sphere*) malloc(sizeof(Sphere) * SPHERES);
	for(int i = 0; i < SPHERES; i++){
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
	}
    
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    
    free(temp_s);
    
    //从球面数据中生成一张位图
    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);
    
    //将位图从GPU复制回到CPU以显示
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    
    
    //获得结束时间，并显示计时结果
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);
    
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));    
    
    //显示位图
    bitmap.display_and_exit();

    //释放内存
    cudaFree(dev_bitmap);
}
```

