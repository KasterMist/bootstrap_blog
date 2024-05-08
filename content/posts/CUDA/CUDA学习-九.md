---
title:       	"CUDA学习(九)"
subtitle:    	""
description: 	""
date:        	2024-03-08
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

这部分是基于原子操作章节进行的高级操作介绍，即实现锁定数据结构。

原子操作只能确保每个线程在完成读取-修改-写入的序列之前，将不会有其他的线程读取或者写入目标内存。然而，原子操作并不能确保这些线程将按照何种顺序执行。例如当有三个线程都执行加法运算时，加法运行的执行顺序可以为(A + B) + C，也可以为A + (B + C)。这对于整数来说是可以接受的，因为中间结果可能被截断，因此(A + B) + C通常并不等于A + (B + c)。因此，浮点数值上的原子数学运算是否有用就值得怀疑🤔。因此在早期的硬件中，浮点数学运算并不是优先支持的功能。

然而，如果可以容忍在计算结果中存在某种程度的不确定性，那么仍然可以完全在GPU上实现归约运算。我们首先需要通过某种方式来绕开原子浮点数学运算。在这个解决方案中仍将使用原子操作，但不是用于算数本身。

<!--more-->

### 原子锁

基本思想是，分配一小块内存作为互斥体，互斥体这个资源可以是一个数据结构，一个缓冲区，或者是一个需要以原子方式修改的内存位置。当某个线程从这个互斥体中读到0时，表示没有其他线程使用这块内存。因此，该线程就可以锁定这块内存，并执行想要的修改，而不会收到其他线程的干扰。要锁定这个内存位置，线程将1写入互斥体，这将防止其他竞争的线程锁定这个内存。然后，其他竞争线程必须等待直到互斥体的所有线程将0写入到互斥体后才能尝试修改被锁定的内存。实现锁定过程的代码可以像下面这样:

```c
void lock(void){
    if(*mutex == 0){
        *mutex = 1; //将1保存到锁
    }
}
```

不过这段代码中存在一个严重的问题。如果在线程读取到0并且还没有修改这个值之前，另一个线程将1写入到互斥体，那么会发生什么情况？也就是说，这两个线程都将检查mutex上的值，并判断其是否为0。然后，它们都将1写入到这个位置，并且都执行后面的语句。这会产生严重的后果。

我们想要完成的操作是：将mutex的值与0相比较，如果mutex等于0，则将1写入到这个位置。要正确实现这个操作，整个运算都需要以原子方式执行，这样就可以确保当线程当线程检查和更新mutex值时，不会有其他的线程进行干扰。在CUDA中，这个操作可以通过函数atomicCAS()来实现，这是一个原子的比较-交换操作(Compare-and-Swap)。函数atomicCAS()的参数包括一个指向目标内存的指针，一个与内存中的值进行比较的值，以及一个当比较相等时保存到目标内存上的值。通过这个操作，我们可以实现一个GPU锁定函数，如下：

```c
__device__ void lock(void){
    while(atomicCAS(mutex, 0, 1) != 0);
}
```

调用atomicCAS()将返回位于mutex地址上的值。**因此while循环会不断运行，直到atomicCAS发现mutex的值为0。当发现为0时，比较操作成功，线程将把1写入到mutex。本质上来看，这个线程将在while循环中不断重复，直到它成功地锁定这个数据结构。**我们将使用这个锁定机制来实现GPU散列表。下面是一种实现方式：

```c
struct Lock{
    int *mutex;
    Lock(void){
        int state = 0;
        HANDLE_ERROR(cudaMalloc((void**)& mutex, sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    ~Lock(void){
        cudaFree(mutex);
    }
    
    __device__ void lock(void){
        while(atomicCAS(mutex, 0, 1) != 0);
    }
    
    __device__ void unlock(void){
        atomicExch(mutex, 0);
    }
}
```

代码中通过atomicExch(mutex, 0)来重制mutex的值，将其与第二个参数进行交换，并返回它读到的值。然而，为什么不用跟简单的方法，例如`*mutex = 0;`呢，因为原子事务和常规的内存操作将采用不同的执行路径。如果同时使用原子操作和标准的全局内存操作，那么将使得unlock()与后面的lock()调用看上去似乎没有被同步。虽然这种混合使用的方式仍可以实现正确的功能，但是为了增加应用程序的可读性，对于所有对互斥体的访问都应使用相同的方式。因此，在使用原子语句来锁定资源后，同样应使用原子语句来解锁资源。



我们想要在最早的点积运算示例中加上原子锁。Lock结构位于lock.h中，在修改后的点积示例中将包含这个头文件。

```c
#include "../common/book.h"
#include "lock.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);
```

核函数也有所不同，在修改后的点积函数中，将Lock类型的变量，以及输入矢量和输出缓冲区传递给核函数。Lock将被用于在最后的累加步骤中控制对输出缓冲区的访问。另一处修改是float *c。之前float *c是一个包含N个浮点数的缓冲区，其中每个线程块都将其计算得到的临时结果保存到相应的元素中。这个缓冲区将被复制到CPU以便计算最终的点积值。然而，现在的参数c将不再指向一个缓冲区，而是指向一个浮点数值，这个值表示a和b中矢量的点积。

```c
__global__ void dot(Lock lock, float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIds.s + blockIdx.x * blockDim.x;
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
    
    //对于归约运算来说，以下代码要求threadPerBlock必须是2的幂
    int i = blockDim.x / 2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    
```

现在到执行到这里时，每个线程块中的256个线程都已经把各自计算的乘积相加起来，并且保存到cache[0]中。现在，每个线程块都需要将其临时结果相加到c执行的内存位置上。为了安全地执行这个操作，我们将使用锁来控制对该内存位置的访问，因此每个线程在更新*c的值之前，要先获取这个锁。在线程块的临时结果与c处的值相加后，将解锁互斥体，这样其他的线程可以继续累加它们的值。在将临时值与最终结果相加后，这个线程块将不再需要任何计算，因此从核函数中返回。

```c
    if(cacheIndex == 0){
        lock.lock();
        *c += cache[0];
        lock.unlock();
    }
}
```

下面是main函数：

```c
int main(){
    float *a, *b, c = 0;
    float *dev_a, *dev_b, *dev_c;
    
    //在CPU上分配内存
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    
    //在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(float)));
    
    //用数据填充主机内存
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }
    
    //将数组“a“和”b“复制到GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    //将“0”复制到dev_c
    HANDLE_ERROR(cudaMemcpy(dev_c, &c, N * sizeof(float), cudaMemcpyHostToDevice));
    
    //声明Lock，调用核函数，并将结果复制回CPU。
    Lock lock;
    dot<<<blocksPerGrid, threadsPerBlock>>>(lock, dev_a, dev_b, dev_c);
    
    //将数组“c”从GPU复制到CPU
    HANDLE_ERROR(cudaMemcpy(&c, dev_a, sizeof(float), cudaMemcpyDeviceToHost));
    
    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));
    
    //释放GPU上的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    //释放CPU上的内存
    free(a);
    free(b);
    
}
```



### 散列表 (Hash Table)

接下来我们将介绍hash table的CPU以及GPU实现。

我们先介绍一下什么是hash table。hash table是一种保存键-值二元组的数据结构，一个字典就可以被视为一个hash table。我们在使用hash table时，需要将查找某个键对应的值的时间降到最低。我们希望这是一个常量时间，即不管hash table多大，搜索某个键对应的值所需时间应该是不变的。

hash table根据值相应的键，把值放入到“桶(bucket)”中。这种将键映射到值的方法通常被称为散列函数(Hash Function)。对于理想的hash function，每个键都会被映射到一个不同的桶。然而，当多个键被映射到同一个桶中时，我们需要将桶中的所有值保存到一个链表里，每当同一个桶添加新的值时，就将新的值添加到链表的末尾。



#### CPU散列表

我们将分配一个长度为N的数组，并且数组中的每个元素都表示一个键-值二元链表。下面是实现的数据结构：

```c
#include "../common/book.h"

struct Entry{
    unsigned int key;
    void* value;
    Entry* next;
};

struct Table{
    size_t count;
    Entry** entries;
    Entry* pool;
    Entry* firstFree;
}
```

Entry结构中包含了键和值。在应用程序中，键是无符号整数。与键相关的值可以是任意数据类型，因此我们将value声明为一个void\*变量。程序重点是介绍如何创建hash table数据结构，因此在value域中并不保存任何内容，仅保证完整性。结构Entry最后的一个成员是指向下一个Entry节点的指针。在遇到冲突时，在同一个桶中将包含多个Entry节点，因此我们决定将这些对象保存为一个链表。所以每个对象都指向桶中的下一个节点，从而形成一个节点链表。最后一个Entry节点的next指针为NULL。

本质上，table结构本身是一个“桶”数组，这个桶数组是一个长度为count的数组，其中entries中的每个桶都是指向某Entry的指针。如果每添加一个Entry节点时都分配新的内存，那么将对程序性能产生负面的影响。为了避免这种情况，hash table将在成员pool中维持一个可用Entry节点的数组。firstFree指向下一个可用的Entry节点，因此当需要将一个节点添加到hash table时，只需使用由firstFree指向的Entry然后递增这个指针，就能避免新分配内存，而且只需要free()一次就能释放所有这些节点。

下面是其他的支持代码：

```c
void initialize_table(Table &table, int entries, int elements) {
    table.count = entries;
    table.entries = (Entry**)calloc( entries, sizeof(Entry*) );
    table.pool = (Entry*)malloc( elements * sizeof( Entry ) );
    table.firstFree = table.pool;
}
```

在hash table初始化的过程中，主要的操作包括为桶数组entries分配内存，我们也为节点池分配了内存，并将指针firstFree初始化为指向节点池数组中的第一个节点。程序末尾释放内存时将释放桶数组和空闲节点池:

```c
void free_table(Table &table){
    free(table.entries);
    free(table.pool);
}
```

在本示例中，我们采取了无符号整数作为键，并且需要将这些键映射到桶数组的索引。也就是说，将节点e保存在table.entries[e.key]中。然而，我们需要确保键的取值范围将小于桶数组的长度。下面是解决方法：

```c
size_t hash(unsigned int key, size_t count){
    return key % count;
}
```

这里的实现方式是将键对数组长度取模，实际情况会更复杂，这里仅仅作为示例程序来展示。我们将随机生成键，如果我们假设随机数值生成器生成的值大致是平均的，那么这个hash function应该将这些键均匀地映射到hash table的所有桶中。真正的实际情况中我们可能需要创建更为复杂的hash function。

将键值二元数组添加到hash table中包括三个基本的步骤:

1. 将键放入hash function中计算出新节点所属的桶。
2. 从节点池中取出一个预先分配的Entry节点，初始化该节点的key和value等变量。
3. 将这个节点插入到计算得到的桶的链表首部。

下面是实现的代码:

```c
void add_to_table(Table &table, unsigned int key, void* value){
    // step 1
    size_t hashValue = hash(key, table.count);
    
    // step 2
    Entry* location = table.firstFree++;
    location->key = key;
    location->value = value;
    
    // step 3
    location->next = table.entries[hashValue];
    table.entries[hashValue] = location;
}
```

步骤3可能会有点难理解。链表的第一个节点是储存在了table.entries[hashValue]中，我们需要在链表的头节点中插入一个新的节点(如果在链表的末尾插入新的节点的话则需要对链表进行遍历直到末尾，增加了复杂度): 首先将新节点的next指针设置为指向链表的第一个节点，然后再将新节点保存到桶数组中(桶数组保存的是链表的第一个节点)，这样就完成了。

为了判断这段代码能否工作，我们实现了一个函数对hash table执行完好性检查。检查过程中首先遍历这张表并查看每个节点。将节点的键放入到hash function进行计算，并确认这个节点被保存到正确的桶中。在检查了每个节点后，还要验证hash table中的节点数量确实等于添加到hash table的节点数量。如果这些数值并不相等，那么要么是无意中将一个节点添加到多个桶，要么没有正确的插入节点。

```c
#define SIZE (100 * 1024 * 1024)
#define ELEMENTS (size / sizeof(unsigned int))

void verify_table(const Table &table){
    int count = 0;
    for(size_t i = 0; i < table; i++){
        Entry* current = table.entries[i];
        while(current != NULL){
            count++;
            if(hash(current->value, table.count) != i){
                printf("%d hashed to %ld, but was located at %ld\n", current->value, hash(current->value, table.count), i);
                current = current->next;
            }
        }
        if(count != ELEMENTS){
            printf("%d elements found in hash table. Should be %ld\n", count, ELEMENTS);
        }
        else{
            printf("All %d elements found in hash table.\n", count);
        }
    }
}
```

由于大部分的功能实现都放到了函数中，因此main()函数就相对比较简单:

```c
#define HASH_ENTRIES 1024

int main(){
    unsigned* buffer = (unsigned int*)big_random_block(SIZE);
    clock_t start, stop;
    start = clock();
    
    Table table;
    initialize_table(table, HASH_ENTRIES, ELEMENTS);
    
    for(int i = 0; i < ELEMENTS; i++){
        add_to_table(table, buffer[i], (void*)NULL);
    }
    
    stop = clock();
    float elapsedTime = (float)(stop - start) / (float)CLOCK_PER_SEC * 1000.0f;
    printf("Time to hash: %3.1f ms\n", elapsedTime);
    
    verify_table(table);
    free_table(table);
    free(buffer);
}
```

我们首先分配了一大块内存来保存随机数值。这些随机生成的无符号整数将被作为插入到hash table中的键。在生成了这些数值后，接下来将读取系统时间以便统计程序的性能。我们对hash table进行初始化，然后通过for循环将每个随机键插入到hash table。在添加了所有的键后，再次读取系统时间，通过之前读取的系统时间与这次系统时间就可以计算出在初始化和添加键上花费的时间。最后，我们通过完整性检查函数来验证hash table，并且释放了分配的内存。



#### 多线程环境下的hash table

多线程环境下的hash table可能会遇到race condition。那么如何在GPU上构建一个hash table呢？在点积示例中，每次只有一个线程可以安全地将它的值与最终结果相加。如果每个桶都有一个原子锁，那么我们可以确保每次只有一个线程对指定的桶进行修改。



#### GPU hash table

在有了某种方法来确保对hash table实现安全的多线程访问，我们就可以实现GPU的hash table的应用程序。我们需要使用Lock，还需要把hash function声明为一个\_\_device\_\_函数。

```c
#include "../common/book.h"
#include "lock.h"

struct Entry{
    unsigned int key;
    void* value;
    Entry* next;
}

struct Table{
    size_t count;
    Entry** entries;
    Entry* pool;
}

__device__ __host__ size_t hash(unsigned int value, size_t count){
    return value % count;
}
```

当\_\_host\_\_与\_\_device\_\_关键字一起使用时，将告诉NVIDIA编译器同时生成函数在设备和主机上的版本。设备版本的函数将在设备上运行，并且只能从设备代码中调用。主机版本的函数将在主机上运行，并且只能从主机代码中调用。\_\_host\_\_与\_\_device\_\_关键字一起使用可以让这个函数既可以在设备上使用又可以在主机上使用。

initialize\_table()和free\_table()与CPU版本差别不大，只是数组的初始化以及释放的代码修改成的GPU版本:

```c
void initialize_table(Table &table, int entries, int elements){
    table.count = entries;
    HANDLE_ERROR(cudaMalloc((void**)&table.entries, entries * sizeof(Entry*)));
    HANDLE_ERROR(cudaMemset(table.entries, 0, entries * sizeof(Entry*)));
    HANDLE_ERROR(cudaMalloc((void**)&table.pool, elements * sizeof(Entry)));
}

void free_table(Table &table){
    cudaFree(table.pool);
    cudaFree(table.entries);
}
```

verify_table()的CPU版本和GPU版本相同，仅仅需要在开头增加一个函数将hash table从GPU复制到CPU。下面是将hash table从GPU复制到CPU的代码:

```c
void copy_table_to_host(const Table &table, Table &hostTable){
    hostTable.count = table.count;
    hostTable.entries = (Entry**)calloc(table.count, sizeof(Entry*));
    hostTable.pool = (Entry*)malloc(ELEMENTS * sizeof(Entry));
    
    HANDLE_ERROR(cudaMemcpy(hostTable.entries, table.entries, table.count * sizeof(Entry*), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(hostTable.pool, table.pool, ELEMENTS * sizeof(Entry), cudaMemcpyDeviceToHost));

```

在复制的数据中，有一部分数据是指针。我们不能简单地将这些指针复制到主机，**因为这些指针指向的地址是在GPU上，它们在主机上并不是有效的指针**。然而，这些**指针的相对偏移量**仍然是有效的。每个指向Entry节点的GPU指针都指向数组table.pool[]中的某个位置，但为了在主机上使用hash table，我们需要它们指向数组hostTable.pool[]中相同的Entry。

给定一个GPU的指针X，需要将这个指针相对于table.pool的偏移与hostTable.pool相加，从而获得一个有效的主机指针，新指针应该按照以下公式计算:
$$
(X - table.pool) + hostTable.pool
$$
对于每个被复制的Entry指针，都要执行这个更新操作：包括hostTable.entries中的Entry指针，以及hash table的节点池中每个Entry的next指针:

```c
    for(int i = 0; i < table.count; i++){
        if(hostTable.entries[i] != NULL){
            hostTable.entries[i] = (Entry*)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
        }
    }
    for(int i = 0; i < ELEMENTS; i++){
        if(hostTable.pool[i].next != NULL){
            hostTable.pool[i].next = (Entry*)((size_t)hostTable.pool[i].next - (size_t)table.pool + (size_t)hostTable.pool);
        }
    }
}
```

在介绍完了数据结构、hash function、初始化过程、内存释放过程以及验证代码后，还剩下的重要部分就是CUDA C原子语句的使用。核函数add\_to\_table()的参数包括一个键数组、一个值数组、hash table本身以及一个lock数组。这些数组将被用于锁定hash table中的每个桶。由于输入的数据是两个数组，并且在线程中需要对这两个数组进行索引，因此还需要将索引线性化:

```c
__global_ void add_to_table(unsigned int* keys, void** values, Table table, Lock* lock){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
```

线程会像点积示例那样遍历输入数组。对于数组key[]中的每个键，线程都将通过hash function计算出这个键值二元数组属于哪个桶。在计算出目标桶之后，线程会锁定这个桶，添加它的键值二元组，然后解锁这个桶。

```c
    while(tid < ELEMENTS){
        unsigned int key = keys[tid];
        size_t hashValue = hash(key, table.count);
        for(int i = 0; i < 32; i++){
            if((tid % 32) == i){
                Entry* location = &(table.pool[tid]);
                location->key = key;
                location->value = values[tid];
                lock[hashVaue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                lock[hashValue].unlock();
            }
        }
        tid += stride;
    }
}
```

for循环和后面的if语句看上去是不必要的。然而，代码中的线程束是一个包含32线程的集合，并且这些线程以步调一致的方式执行。每次在线程束中只有一个线程可以获取这个锁。如果让线程束中的所有32给线程都同时竞争这个锁，那么将会发生严重的问题。这种情况下，最好的方式就是在软件中执行一部分工作，遍历线程束中的线程，并给每个线程一次机会来获取数据结构的锁，执行它的工作，然后释放锁。

main函数的执行流程跟CPU版本的相似:

```c
int main(){
    unsigned int* buffer = (unsigned int*)big_random_block(SIZE);
    
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    unsigned int* dev_keys;
    void** dev_values;
    HANDLE_ERROR(cudaMalloc((void**)&dev_keys, SIZE));
    HANDLE_ERROR(cudaMalloc((void**)&dev_values, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_keys, buffer, SIZE, cudaMemcpyHostToDevice));
    
    Table table;
    initialize_table(table, HADH_ENTRIES, ELEMENTS);
    
    // 声明一个锁数组，数组中的每个锁对应于桶数组中的每个桶。并将它们复制到GPU上。
    Lock lock[HASH_ENTRIES];
    Lock* dev_lock;
    HANDLE_ERROR(cudaMalloc((void**)&dev_lock, HASH_ENTRIES * sizeof(Lock)));
    HANDLE_ERROR(cudaMemcpy(dev_lock, lock, HASH_ENTRIES * sizeof(Lock), cudaMemcpyHostToDevice));
    
    // 将键添加到hash table，停止性能计数器，验证hash table的正确性，执行释放工作
    add_to_table<<<60, 256>>>(dev_keys, dev_values, tavle, dev_lock);
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to hash: 3%3.1f ms\n", elapsedTime);
    
    verify_table(table);
    
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    free_table(table);
    cudaFree(Dev_lock);
    cudaFree(dev_keys);
    cudaFree(dev_values);
    free(buffer);
}
```

