---
# type: docs 
title: Iso_suface
date: 2024-06-18T15:28:33+08:00
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



<!--more-->

## iso\_surface代码结构:

- evaluator.cpp / evaluator.h
- hash\_grid.cpp / hash\_grid.h
- index.h
- iso\_method\_ours.cpp / iso\_method\_ours.h
- iso\_surface\_generator.cpp / iso\_surface\_generator.h
- mesh.cpp / mesh.h
- multi\_level\_researcher.cpp / multi\_level\_research.h
- qefnorm.h
- surface\_reconstructor.cpp / surface\_reconstructor.h
- traverse.h
- visitorextract.cpp / visitorextract.h



## 代码细节

### evaluator

用来初步处理global particles，得到一系列的Global数据

##### singleEval()

##### singleEvalWithGrad()

##### GridEval()

...

计算梯度

##### CheckSplash()

##### CalculateMaxScalarConstR()

##### CalculateMaxScalarVarR

##### RecommendIsoValueConstR()

##### RecommendIsoValueVarR()

##### CalcParticlesNormal()

##### general_kernel()

##### gradient_kernel()

##### AnisotropicInterpolate()

##### AnisotropicInterpolateGrad()

##### compute_xMeans()

##### compute_G()

##### compute_Gs_xMeans()

##### wij()

### hash\_grid



### index



### iso\_method\_ours

包含了TNode(Tree node) struct的实现

#### TNode

##### 变量

- SurfReconstructor* constructor;
- Eigen::vector3f center;
- float half\_length;
- Eigen::vector4f node = Eigen::Vector4f::Zero(); **--> ???**
- unsigned long long id; TNode的id
- std::array\<std::shared\_ptr\<TNode\>, 8\>; TNode的children



##### TNode(SurfReconstructor* surf\_constructor, unsigned long long id)

对TNode进行初始化，children都设置为0, 赋值id

##### TNode(SurfReconstructor* surf\_constructor, std::shared\_ptr\<TNode\> parent, Index i)

对TNode进行初始化，对children设置为0的同时，也根据parent信息设置了depth, node, id.

##### float calcErrorDMC()

**???**

##### void GenerateSampling()

**???**

##### void NodeSampling()

**???**

##### void Node CalcNode()

**???**

##### bool changeSignDMC()

**???**



### iso\_surface\_generator

根据xmf文件生成isosurface

这个生成的isosurface格式是什么，包含了哪些信息？生成的isosurface格式是surfReconstructor?

### mesh

包含了vect3 struct, Triangle struct, Mesh class的实现。

这个mesh指的是什么？和surface有什么不同

### multi\_level\_researcher



### qefnorm



### surface\_reconstructor

#### SurfReconstructor

包含了SurfReconstructor class的实现。包含了很多Global Parameters

**这个SurfReconstructor是什么？**

##### loadRootBox()

根据\_GlobalParticles的点，对\_BoundingBox进行设置最大值和最小值。即通过\_GlobalParticles点更新\_BoundingBox范围

##### generalModeRun()

执行一整套运行流程

##### shrinkBox()

**???**

##### resizeRootBoxConstR()



##### resizeRootBoxVarR()



##### genIsoOurs()



##### checkEmptyAndCalcCurv()



##### beforeSampleEval()



##### afterSampleEval()



### traverse



### visitorextract



## 项目流程

开会讨论marching cubes的实现细节

- project中使用iso\_surface的地方
- 询问有没有实际的测试用例
- 寻找提高并行效率的地方
- CPU、GPU
- bottom up的可能性
- 是否需要重构八叉树，把cornerstone octree复现到project里面



**使用流程**

第一步: global particle --> Evaluator class

第二步：evaluator class --> SurfReconstructor class

第三步: 



// 构建八叉树停止规则: 等直面平了，节点到了粒子半径的时候



能否将cornerstone整体搬到project里面，先bottom up构建树，然后再进行进行剪枝。**在此project中，剪枝需要进行大量的数据计算。**



idea1: 使用cornerstone得代码进行bottom up构建树。可以并不采用multi-processes构建，可以单进程构建，因为相比构建树，剪枝需要大量的运算。构建完树之后可以并行的进行剪枝。树的类型是Octree?Csarray?



### 实际样例: 

1. 读取文件
2. 创建ISOSurfaceGenerator class并调用generate()函数
3. 如果并不存在需要生成的ply文件:
   1. 创建SurfReconstructor并初始化
   2. 读取h5文件
   3. 判断IS\_CONST\_RADIUS，此样例为True，即只用一个radius数值。调用对应的SurfReconstructor构造函数
   4. SurfReconstructor::setSplashFactor()
   5. SurfReconstructor::Run()执行generalModeRun()
      1. SurfReconstructor::loadRootBox(): 根据\_GlobalParticles的点，对\_BoundingBox进行设置最大值和最小值
      2. IS\_CONST\_RADIUS --> 创建一个HashGrid对象并初始化和赋值
      3. 创建Evaluator对象并初始化和赋值
      4. Evaluator::compute\_Gs\_xMeans()
      5. IS\_CONST\_RADIUS --> resizeRootBoxConstR()
      6. 这时已经确认了max depth和min depth **???**
      7. IS\_CONST\_RADIUS --> Evaluator::CalculateMaxScalarConstR()
      8. IS\_CONST\_RADIUS --> Evaluator::RecommendIsoValueConstR()
      9. CALC\_P\_NROMAL --> Evaluator::CalcParticlesNormal()
      10. genIsoOurs()执行两次
   6. 将计算结果写入ply文件



一些问题: 

1. SurfReconstructor::setSplashFactor()的功能
2. SurfReconstructor::Run()功能
   1. SurfReconstructor::loadRootBox():
   2. Evaluator::compute\_Gs\_xMeans():
   3. resizeRootBoxConstR(): 
   4. 确认了max depth和min depth?
   5. Evaluator::CalculateMaxScalarConstR()
   6. Evaluator::RecommendIsoValueConstR()
   7. Evaluator::CalcParticlesNormal()
3. tree是在哪一步开始建立的
4. genIsoOurs()的功能和作用
5. constructor->Run()后writePlyFile(mesh, ply\_path), 说明了constructor->Run()修改了mesh信息?



瓶颈问题:

第一步自上而下建树(只考虑空间因素的八叉树)，只考虑空或者非空的情况。为了剪枝操作。其中对每个节点的采样需要大量计算。

第二步自下而上的剪枝。
