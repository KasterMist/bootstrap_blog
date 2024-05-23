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

### tree

#### csarray.hpp

##### findSearchBounds()

确定搜索范围，用于在八叉树中查找包含目标区域的节点。该函数通过比较目标区域和节点的边界，返回包含该区域的节点索引。

##### updateNodeCount()

递归遍历节点树，计算并更新每个节点及其子节点中的粒子总数。

##### computeNodeCounts()

计算每个节点及其子节点中的总粒子数。返回每个node的粒子数。

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

遍历八叉树，对每个节点的状态和数据进行更新，确保树结构的一致性和正确性。

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

## test

### performance

#### octree.cpp

通过numParticles和bucketSize(每个node最大particles数量)生成随机的高斯坐标。之后使用`build_tree()`构造树，`halo_discovery()`查找halo，`computeSfcKeys()`计算sfc keys，根据sfc keys大小重新调用`build_tree()`

`RandomGaussianCoordinates`在`test/coord_samples/random.hpp`里面。

##### build_tree()

`computeOctree()`: 在`include/cstone/tree/csarray.hpp`里。

`updateOctree()`

##### halo_discovery()



