---
# type: docs 
title: Cornerstone代码解析
date: 2024-05-21T17:03:31+08:00
featured: false
draft: true
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: []
tags: []
images: []
---

git链接: https://github.com/sekelle/cornerstone-octree.git

<!--more-->

粒子是怎么生成的？

如何赋值到各自的节点上去的？

树的生成方法？

decomposition的规则是什么？

怎么rebalance tree的？如果新加了点之后如何rebalance？



# 结构

## include/cstone

### domain

**注: 如果domain设置tag为GPU的话，里面的一些变量会存储到device里面，访问时需要注意！！！**

#### domain.hpp

domain class用来管理分布式的particles以及halos (用于多进程，多GPU的情况)

domain对象创建的时候需要rank, nRanks, bucketSize, bucketSizeFocus, theta, box作为输入:

- rank: mpi rank值
- nRank: mpi rank总数
- bucketSize: maximum particles per node in the global tree. 需要大于bucketSizeFocus
- bucketSizeFocus: 在assigned part中最大的particles per leaf node
- theta: 控制焦点分辨率和重力精度的角度参数
- box: global bounding box

初始化时会初始化myRank\_(rank), numRanks\_(nRanks), bucketSizeFocus_(bucketSizeFocus), theta\_(theta), focusTree\_(rank, numRanks\_, bucketSizeFocus\_), global\_(rank, nRanks, bucketSize, box)

这个global\_可以视为一个GlobalAssignment类的对象 **???**，存储着global信息以及进行数据分发。



##### sync() 更新particles的顺序(与xyz坐标,radius,以及其属性有关)

输入:

- x, y, z坐标vector (locally distributed)
- h radius
- scrathbuffers: 暂存缓冲器，需要至少三个vector，两个用来send/receive，一个用来存储SFC order

输出

- particleKeys: 输出一个存储了排好顺序的particleKeys的vector
- x, y, z, h的vector会根据h的信息进行更新，添加halo。更新后的x, y, z, h会包含各自rank要处理的点+halo的点。
- particleProperties

domain对象中下面的信息将被更新:

- global tree: Update of the global octree, for use as starting guess in the next call  **???**
- assigned range startIndex(), endIndex(): 可以理解为最初的domain里面的点是无序的，后面变成有序的，通过startIndex()和endIndex()来将particles分配给不同的进程
- total local perticle count (assigned + halo particles)
- halo exchange pattern: **???**
- global coordinate bounding box: **???**

###### 流程

1. staticChecks(): 对暂存缓冲器进行检查，判断格式、数量是否符合要求。
2. 之后将scratchbuffers分为scratch(除了最后一个vector)，和sfcOrder(最后一个vector)，来处理不同的任务。
3. auto [exchangeStart, keyView] = distribute()  **???**
   1. initBounds: 设置buffersize
   2. global\_  这个变量的含义是什么 **???**
4. gatherArrays() **???**
5. findPeersMac() 根据MAC和dual traversal找到当前进程的peer ranks(可以理解为临近的进程)。使用的是global data，并不需要communication
6. 对focusTree\_进行处理  
   1. 如果是第一次sync，需要调用focusTree\_.converge(): 
      1. 对每个rank的focustree进行update，并使用MPI\_Allreduce()来判断是否所有的rank的focustree已经converged
   2. focusTree\_.updateMinMac(): updateMAC criteria(based on a min distance MAC)
   3. focusTree\_.updateTree(): update focusTree的结构
   4. focusTree\_.updateCounts(): **global update** of the tree structure, 可能是将focusTree的信息更新到global tree上面  **!!!**
   5. focusTree\_.updateGeoCenters(): 更新vector，比如扩大size
7. updateLayout() 对layout进行处理。**layout是什么.**  **???**
8. setupHalos()



##### focusTree --> FocusedOctree Class

locally focused, fully traversable octrree, used for halo discovery and exchange.

根据focused SFC area建立focusTree

###### 初始化:

focusTree\_(rank, numRanks\_, bucketSizeFocus\_)

会根据GPU或CPU进行不同的初始化处理

如果是GPU，会在GPU上进行构建focusTree --> memcpyH2D()

如果是CPU，会在CPU上进行构建focusTree --> updateInternalTree()

###### focusTree.converge()

对各自rank的focusTree进行更新，直到所有rank的focusTree全部更新完。



#### assignment.hpp

implementation of global particle assignment and distribution

##### assign()

对global tree进行更新

##### distribute()

根据assignment情况，将particles分发给其他rank

里面的`exchangeParticles()`涉及到了MPI的函数 (在domaindecom\_mpi.hpp里面)



### primitives

#### gather

GpuSfcSorter类，用于在GPU上对SFC码进行排序，并生成重新排序的映射表。



### sfc

#### sfc.hpp

包含了将坐标转化为sfc3D,sfc2D的功能，以及根据sfc类型(morton, hilbert)进行decode。可参考hilbert.hpp, morton.hpp.

#### hilbert.hpp

包含了hilbert的encode和decode

#### morton.hpp

包含了morton的encode和decode



### tree

#### octree.hpp

##### binaryKeyWeight()

计算二进制节点索引(binary node index)到八叉树节点索引(octree node index)的映射。

##### createUnsortedLayoutCpu()

将内部节点和叶子树合并成一个带有nodeKey前缀的数组。

##### linkTreeCpu()

从二进制树结构中提取父子关系，并转换为排序后的八叉树节点顺序。

##### getLevelRangeCpu()

确定八叉树各级节点的范围和边界。

##### buildOctreeCpu()

在CPU上通过递归算法构建八叉树的内部结构，包括节点分割和链接。

##### locateNode()

根据SFC key range[startKey:endKey]查找是否有对应node，有的话返回这个node的index  

##### containingNode()

查找包含特定键的最小节点。

##### maxDepth()

返回树的最大深度。

##### updateInternalTree()

更新树的结构。

##### leafToInternal()

提供叶节点与其对应的内部节点之间的映射关系。

#### csarray.hpp --> 根据cornerstone格式生成local和global octrees, 包含了cornerstonre octree core的函数

##### findSearchBounds()

给定一个particle sfc code值，查找其对应的范围区间

##### updateNodeCount()

对每一个node进行更新对应的count。通过给定当前的node值(即当前node包含的第一个particle sfc code值)和下一个node的值，来确定count数量。

##### computeNodeCounts() !

给定所有的坐标点的sfc codes，tree的基本信息，计算每个octree node的粒子数量。

##### siblingAndLevel()

计算节点的同级兄弟节点和层级信息。返回一个node index(如果没有sibling，则返回-1)和一个tree level index

##### calculateNodeOp()

根据节点的粒子数量和其他统计信息，计算该节点需要执行的操作数。

##### rebalanceDecision()

判断是否需要rebalance，返回bool值。

##### processNode()

处理单个节点的操作，如拆分或者合并(基于nodeOps的值)。

##### rebalanceTree()

通过递归算法，对树结构进行重新平衡，确保每个节点的负载均衡。

`exclusiveScan()`计算数组的前缀和

for循环`processNode()`

##### updateOctree()

遍历八叉树，对每个节点的状态和数据进行更新，确保树结构的一致性和正确性。返回两个vector值: 

1. tree: the octree leaf nodes (cornerstone format)
2. counts: the octree leaf node particle count 每个叶子节点的粒子数量

`rebalanceDecision()`判断是否需要rebalance

`rebalanceTree()`进行rebalance

`computeNodeCounts()`tree balance过后每个node的partical counts会发生变化，需要进行计算。

##### computeOctree()

初始化八叉树的根节点，递归地计算和设置每个子节点的状态和数据。

循环调用`updateOctree`直到Octree为最新。

##### updateTreelet()

基于node counts针对局部子树进行遍历和更新操作，确保子树的信息与整体八叉树一致。

##### computeSpanningTree()

构建一个包含所有计算节点的跨节点树，用于在分布式系统中进行高效的数据处理和通信。



### util

#### reallocate

##### reallocateDevice(Vector& vector, size_t size, double growthRate)

在需要时重新分配设备向量(vector)的存储空间，以适应新的大小，并按给定的增长率增加容量。用于向cuda设备的vector进行充分配。

##### reallocateGeneric(Vector& vector, size_t size, double growthRate)

CPU的向量重分配。

##### reallocate(Vector& vector, size_t size, double growthRate)

根据vector是否在device中进行选择调用reallocateDevice或reallocateGeneric



### findneighbors.hpp

##### findneighbors(无 LocalIndex i)

根据domain的startIndex和endIndex(即每个进程rank各自分配的domain里面的点的起始位置到结束位置)，对分配的所有的点点进行查找neighbors(neighbors的点是global的，即查找global data里面某个点的临近的点)



##### findneighbors(有 LocalIndex i)

在global data里面查找某个点的临近的点，findneighbors(无 LocalIndex i)会并行调用这个函数。



两种findneighbors都会将某个点的临近neighbors的数量以及所有临近neighbors的index存储起来

通过查找该点对应的tree node来对tree node里面的点进行neighbors查找(使用traverse来对每个searchBox进行查找)

searchBox: 包含一个leaf node的所有particles



## test

### performance

#### octree.cpp

通过numParticles和bucketSize(每个node最大particles数量)生成随机的高斯坐标。之后使用`build_tree()`构造树，`halo_discovery()`查找halo，`computeSfcKeys()`计算sfc keys，根据sfc keys大小重新调用`build_tree()`

`RandomGaussianCoordinates`在`test/coord_samples/random.hpp`里面。

##### build_tree()

`computeOctree()`: 在`include/cstone/tree/csarray.hpp`里。

`updateOctree()`

##### halo_discovery()



### integration_mpi

#### assignment_gpu.cpp --> 对vector进行重分配

##### initCoordinates()

根据mpi rank值生成不同的随机三维坐标

##### randomGaussianAssignment()

`initCoordinates()`: 对x,y,z数组进行随机数赋值，然后将数据分布到box里面。

创建GlobalAssignment对象。GlobalAssignment类用于在GPU上进行全局域分配和粒子分发，利用Thrust库和CUDA进行并行计算(Thrust库是一个用于CUDA编程的C++模板库，提供了一些高层次的并行算法接口)。

创建一个GpuSfcSorter对象，用于在GPU上对SFC码进行排序，并生成重新排序的映射表。

通过GlobalAssignment.assign()获取buffer size

使用reallocate和reallocateDevice对host和device的vector进行重分配。

#### box_mpi.cpp --> 创建globalbox，然后检查正确性🧐

##### TEST(GlobalBox, localMinMax)

检查一个`MinMax`函数模板对一个包含1000个`double`类型元素的向量进行最小值和最大值计算的正确性。

##### void makeGlobalBox(int rank, int numRanks)

创建一个`Box`对象(global box，每个进程都有)，并检查其属性设置的正确性(xmin,xmax,...)，检测在不同的周期性边界条件（PBC）下验证`Box`对象的正确性。

##### TEST(GlobalBox, makeGlobalBox)

测试`makeGlobalBox()`在float和double数据类型下以及不同进程数量情况下的正确性。

#### domain_2ranks.cpp --> 使用两个进程，测试domain创建的情况

domain的定义是什么，到底是什么东西？可以理解为是particles的集合吗，或者说可以理解为一个particles所在范围的区间？

##### void noHalos(int rank, int numRanks)

测试nohalos的情况下不同进程分配不同的domain信息以及domain.sync()的效果

最开始每个进程domain的x,y,z坐标都为{0.5, 0.6} (每个进程包含两个坐标点，总共四个坐标点)

使两个进程进行domain.sync()后

rank0进程的domain的x,y,z坐标都为{0.5, 0.5}，rank1进程的domain的x,y,z坐标都为{0.6, 0.6}

##### TEST(FocusDomain, noHalos)

对`void noHalos(int rank, int numRanks)`进行不同数据类型的测试

##### void withHalos(int rank, int numRanks)

测试存在halos的情况下不同进程分配不同的domain信息以及domain.sync()的效果

每个进程domain的x,y,z坐标都为{0.5, 0.6}，点的半径为{0.2, 0.22}

使两个进程进行domain.sync()后

rank0和1进程的domain的x, y, z坐标为{0.5, 0.5, 0.6, 0.6}, 点的半径为{0.2, 0.2, 0.22, 0.22}

区别在于rank0的startindex和endindex为0和2，rank1的startindex和endindex为2和4.

##### TEST(FocusDomain, halos)

对`void withHalos(int rank, int numRanks)`进行不同数据类型的测试

##### void moreHalos(int rank, int numRanks)

##### void particleProperty(int rank, int numRanks)

##### void multiStepSync(int rank, int numRanks)

##### void zipSort(std::vector<T>& x, std::vector<T>& y)

##### void domainHaloRadii(int rank, int nRanks)

#### domain_gpu.cpp

注: 里面的RandomCoordinates generation是每个rank各自生成各自的点。

##### TEST(DomainGpu, matchTreeCpu)

测试CPU和GPU版本的domain.sync是否得到相同的结果。

##### TEST(DomainGpu, reapplySync)

测试GPU条件下reapplySync是否正确

#### exchange\_domain\_gpu.cpp

##### TEST(GlobalDomain, exchangeAllToAll)

测试exchangeParticlesGPU的效果。

exchangeAllToAll会让每个rank的data保留1/numrank，剩下的分发给其他的rank。

exchangeParticles()就是使用mpi进行particles传递。

exchangeParticlesGPU()在exchangeParticles()基础上加上了从device上传递数据回host，再通过host进行communicate数据，最后再将更新的数据从host传递到device上。mpi\_cuda的send和recv函数在primitives/mpi\_cuda.cuh上。

**注:有一种GPU-direct的方式传输，可能是GPU之间直接进行通信**

#### exchange\_halos\_gpu.cpp

##### TEST(HaloExchange, gpuDirect)

测试使用GPU\_direct的情况下MPI直接对GPU的信息进行传递是否能够成功。

##### TEST(HaloExchange, simpleTest)

检测使用haloExchangeGpu()是否能正确的交换halo particles信息。

#### treedomain.cpp



## 疑问

morton和hilbert在什么时候开始使用？double coordinates能否也可以使用morton和hilbert？数据精度丢失怎么办？



/test/performance/octree.cpp  --> computeSfckeys()函数的功能以及之后的使用

--> 可以看一下 /test/unit/sfc/sfc.cpp, morton.cpp, hilber.cpp来查找computeSfckeys()后如何进行进一步处理

/test/integration_mpi/assignment_gpu.cpp  --> 昨天下午一起商讨过

/test/unit/tree/octree.cpp, csarray.cpp

/test/unit/traversal/里面的遍历功能



在/test/performance/octree.cpp中已经计算了computeSfckeys()，但是并没有对return值做进一步处理。其他地方调用了computeOctree，里面的codeStart和codeEnd类型是local particle SFC codes start and end，但是代码里面是直接通过RandomGaussian生成的。那么这种SFC codes与computeSfckeys()生成的结果的区别是什么？

computeSfckeys()为什么要跟box有关联？double类型的点怎么转化为sfc keys？

生成了Octree实例后，是否需要进行link？

如果查找某个点在octree的位置是通过每个leaf node的count数量，然后累加count直到该点的index等于当前count数量的话，是不是意味着在最开始存储点的时候就需要将点进行排序？三维坐标系的点该如何排序？通过sfc(morton code)吗？



### /sample_test/main.cpp

RandomGaussianCoordinates生成randomBox --> 给定box值，包含坐标点的值以及对应的sfc codes

computeOctree() --> 返回tree和counts

1. computeOctree (updateOctree)
   1. rebalanceDecision
   2. rebalanceTree
   3. computeNodeCounts
      1. updateNodeCounts 对每一个node进行更新对应的count。通过给定当前的node值(即当前node包含的第一个particle sfc code值)和下一个node的值，来确定count数量。
         1. findSearchBounds 给定一个particle sfc code值，查找其对应的范围区间

domain.sync?

多节点(多进程)如何对octree做rebalance

findneighbors() 是如何进行查找的，多进程的情况下如何进行并行查找？



### 基于代码使用时产生的疑问

##### 是否每个节点上有完整的global tree？ 

GLobalAssignment类在assignment.hpp里面，功能是对global domain进行配置和分发，同时也创建了一个"low-res" global tree，可以理解为是一个简单的global tree，里面的具体细节和架构在其他的rank里面。

assignment.hpp里面有对于tree\_ (Octree类)的更新，使用的是computeSpanningTree创建一个简单的global tree.

assignment.hpp里面的assign函数调用了updateOctreeGlobal()，这个函数只是将各自rank进行updateOctree或computeNodeCount，然后使用MPI\_Allreduce对counts (leaf node particle counts)进行汇总，没有构造global tree的结构。

**所以，可以视为每个rank上有完整的global tree，但是只是一种非常简单的结构，每个rank有各自的focustree的具体的结构。可以理解为每个rank上有各自的focustree，而这个global tree只起到一个整体作用，细节还是展现在了各个rank来的focustree里面。**

##### halo exchange的功能？

更新focustree的halo信息。

##### focustree update是如何实现的？是重新创建树吗？

实现在octree\_focus\_mpi.hpp里面。会根据peer rank的change以及自身的点的变化来update focustree。focustree是high resolution的，意思是如果某个rank k有sfc range F，那么其他rank不可能会有属于sfc range F且rank k没有的信息。

##### 如果一直往一个区域加点，会有平衡问题吗？

从project的代码来看，tree的update需要输入local particle SFC codes start和end，所以update的时候需要reorder才行。从流程来看，如果往一个区域加点可以先使用domain进行汇总然后sync，之后创建的focustree就会平衡。不过目前看来octree对于增删比较花时间，尽量少用。

##### focusTree和globalTree分别起什么作用？

global tree只起到一个空间作用，真正具体的结构和包含的信息在各自rank的focus tree里面。

##### findNeighbors()如果点是在边界该如何处理？每个focustree包含了临近rank的focustree的部分信息吗

findNeighors()会根据focustree.octreeProperties.nsView()的信息进行查找临近的点。

focusTree.updateTree()会根据peerRanks lists来进行调用updateFocus()来update

focusTree.updateTree() 

1. 更新focusStart和focusEnd
2. 调用focusTransfer更新enforcedKeys(buffer数据)
   1. 根据prevFocusStart和prevFocusEnd与newFocusStart和newFocusEnd的不同来更新tree
   2. 将更新后的tree与更新前的tree的不同部分send给对应的临近rank
   3. mpi\_recv的信息(新的range部分)会被存储到enforcedKeys(buffer数据)里面
3. 调用updateFocus来根据enforcedKeys来更新focusTree

由于domain.sync()更新的x,y,z,h是各自的点+基于h而决定的halo的点，所以边界处理就可以正常处理。所以每个focustree包含了临近rank的focustree的部分信息，即halo部分的信息。

##### halos\_discover() 是如何进行获取halos的，以及如何send halos的信息，send halo的信息是所有的点的信息吗？

通过halos\_.discover()调用findHalos()查找halos。(会根据radius)

- findHalos会对一个collisionFlags进行处理，collisionFlags存储的是当前focustree的的每个leafnode是否为halo的信息(不是记为0，是记为除0之外的数)。视为更新halos\_.collisionFlags\_的信息

send halos是通过computeLayout()进行。

- 通过computeNodeLayout()计算focus tree的每个leaf node的location(offset)
- 通过layout计算newParticleStart和newParticleEnd。
- 计算send和recv的具体信息(mpi的properties)

exchangeHalos()会调用haloExchange()来进行mpisend(Isend)和mpirecv(Irecv)，最后返回一个particleBufferSize

domain.sync()里面之后会调用setupHalos()，其中就有exchangeHalos的调用。

综上，halo的信息可以理解为是所有particle的信息。**具体详细流程分析起来过于复杂，如果有时间可以继续深入研究。**

