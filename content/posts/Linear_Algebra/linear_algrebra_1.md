---
# type: docs 
title: 线性代数相关知识点
date: 2024-07-08T17:25:26+08:00
featured: false
draft: false
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Note"]
tags: ["Math"]
images: []
---

这部分将记录线性代数的相关知识点，简单用于复习以便工作中使用。

<!--more-->

## 基本概念

### 行列式 (Determinant)

行列式是与方阵（即行数和列数相同的矩阵）相关的一个标量值。

**定义**

对于一个$n \times n$的方阵$A$，行列式记作$det(A)$或$|A|$。行列式的具体计算方法随矩阵的维数变化而不同。

- $1\times 1$矩阵: 对于 $A = [a]$，其行列式为$det(A) = a$.
- $2 \times 2$矩阵: 对于 $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$，其行列式为$det(A) = ad - bc$.
- $3 \times 3$矩阵: 对于$A = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$，其行列式为$det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)$.

对于更高维的矩阵，行列式可以通过递归展开，或使用高斯消元法将矩阵转化为上三角矩阵，再计算对角线元素的乘积。

**行列式的性质**

- 如果行列式为0，矩阵不可逆。
- 行列式反映了线性变换的体积缩放因子和方向(正或负)。
- 行列式在行交换时会改变符号，在行或列加倍时也会相应改变。



### 特征值 (Eigenvalue)

#### 特征值的定义

设$A$是一个$n \times n$的方阵。如果存在一个非零向量$v$和一个标量$\lambda$，使得$Av = \lambda v$，那么标量$\lambda$称为矩阵$A$的特征值，对应的非零向量$v$称为特征向量。

在$Av = \lambda v$中，$Av$表示矩阵$A$作用于向量$v$，结果是向量$v$仅仅被拉伸或缩短，而没有改变方向。这个拉伸或缩短的比例因子就是特征值$\lambda$.



#### 求特征值

求解特征值的过程通常涉及以下步骤:

1. 特征多项式: 计算矩阵A得特征多项式。这个多项式通过求解下面的特征方程得到:
   $$
   AI = \lambda I \\
   A - \lambda I = 0 \\
   det(A - \lambda I) = 0
   $$
   $I$是$n \times n$的单位矩阵，$det$表示行列式。

2. 求根: 特征多项式是关于$\lambda$的n次多项式，其根就是矩阵$A$的特征值。

   例子:一个简单的$2 \times 2$矩阵:
   $$
   A = 
   \begin{pmatrix}
   	4 & 1 \\
   	2 & 3 	
   \end{pmatrix}
   $$
   下面求解其特征值:

   1. 构建特征方程:
      $$
      det(A - \lambda I) = \\det \begin{pmatrix}   4 - \lambda & 1 \\   2 & 3 - \lambda   \end{pmatrix} = (4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0   \\
      $$
   2. 求解这个特征多项式:
   $$
   \lambda ^ 2 - 7 \lambda + 10 = 0
   $$
   ​	 我们可以得到特征值:
   $$
   \lambda_1 = 5, \quad \lambda_2 = 2
   $$
   ​	 对于$\lambda = 5$: 
   $$
   \begin{pmatrix} 
   	4 - 5 & 1 \\
   	2 & 3 - 5
   \end{pmatrix} 
   \begin{pmatrix}
   	v_1 \\ 
   	nv_2 
   \end{pmatrix} 
   = 
   \begin{pmatrix} 
   	-1 & 1 \\
   	2 & -2 
   \end{pmatrix} 
   \begin{pmatrix} 
   	v_1 \\ 
   	v_2 
   \end{pmatrix} 
   = 
   \begin{pmatrix} 
   0 \\
   0 
   \end{pmatrix}
   $$
   ​	我们得到$-v_1 + v_2 = 0$，即$v_2 = v_1$。因此，对应于$\lambda = 5$的特征向量是$v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.

​		 对于$\lambda = 2$:
$$
\begin{pmatrix} 
	4 - 2 & 1 \\
	2 & 3 - 2
\end{pmatrix} 
\begin{pmatrix}
	v_1 \\ 
	nv_2 
\end{pmatrix} 
= 
\begin{pmatrix} 
	2 & 1 \\
	2 & 1 
\end{pmatrix} 
\begin{pmatrix} 
	v_1 \\ 
	v_2 
\end{pmatrix} 
= 
\begin{pmatrix} 
0 \\
0 
\end{pmatrix}
$$
​		我们得到$2v_1 + v_2 = 0$，即$v_2 = -2v_1$。因此，对应于$\lambda = 2$的特征向量是$v = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$.





### 矩阵的秩 (rank)

矩阵的秩是指矩阵中线性无关行或列的最大数量



#### 计算矩阵的秩

1. **行简化（Row Reduction）**：通过高斯消元法（Gaussian Elimination）或行最简形式（Row Echelon Form），可以将矩阵转换成上三角矩阵或阶梯形矩阵。非零行的数量就是矩阵的秩。
2. **极大无关子集（Maximal Independent Subset）**：寻找矩阵中最大线性无关的行或列集。
3. **行列式（Determinants）**：对于方阵（即 $n \times n$ 矩阵），如果行列式不为零，则矩阵的秩是 $n$。如果行列式为零，可以通过计算子矩阵的行列式来确定秩。
4. **奇异值分解（Singular Value Decomposition, SVD）**：通过SVD分解矩阵，非零奇异值的数量就是矩阵的秩。

**例子**:

给定一个矩阵A:
$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
$$
通过高斯消元法:
$$
\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix} \xrightarrow{\text{R2} - 4\text{R1}} \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 7 & 8 & 9 \end{pmatrix} \xrightarrow{\text{R3} - 7\text{R1}} \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & -6 & -12 \end{pmatrix} \xrightarrow{\text{R3} - 2\text{R2}} \begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \\ 0 & 0 & 0 \end{pmatrix}
$$

行简化后得到的矩阵有两个非零行，因此矩阵$A$的秩为2。



#### 秩亏矩阵 (rank-deficient)

秩亏矩阵指的是那些秩小于其最大可能值的矩阵。具体来说：

- 对于一个$m \times n$的矩阵$A$，其最大可能秩是 $min(m,n)$。
- 如果矩阵$A$的秩 $𝑟 < min⁡(𝑚,𝑛)$，那么 $A$ 就被称为秩亏矩阵。

秩亏矩阵的特性

1. **行列式为零**：如果一个方阵是rank-deficient的，它的行列式为零。
2. **零特征值**：如果一个矩阵$A$是rank-deficient的，它的特征值分解中会包含至少一个零特征值。
3. **线性相关性**：rank-deficient矩阵的行或列之间存在线性相关性。



#### 奇异值分解（Singular Value Decomposition, SVD）

奇异值分解是将任意的$m \times n$矩阵$A$分解为三个矩阵乘积的方法。这三个矩阵分别是一个正交矩阵、一个对角矩阵和另一个正交矩阵。

**奇异值分解的定义**

对于一个$m * n$的矩阵$A$，其奇异值分解表示为:
$$
A = U\Sigma V^T
$$
其中：

- $U$是一个$m \times m$的正交矩阵，成为左奇异矩阵。
- $\Sigma$是一个$m \times n$的对角矩阵，成为奇异值矩阵，其对角元素是$A$的奇异值。
- $V$是一个$n \times n$的正交矩阵，成为右奇异矩阵。

**奇异值分解的性质**

- 奇异值是非负的实数，排列在对角矩阵$\Sigma$的对角线上。
- 奇异值的数量等于矩阵$A$的秩。
- 奇异值分解用于降维、数据压缩和噪声过滤。
- 奇异值分解还用于主成分分析(PCA)和最小二乘问题的求解。 
