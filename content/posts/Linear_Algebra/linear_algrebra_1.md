---
# type: docs 
title: 线性代数相关知识点
date: 2024-07-08T17:25:26+08:00
featured: false
draft: true
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Note"]
tags: ["Math"]
images: []
markup: mmark
---

这部分将记录线性代数的相关知识点，简单用于复习以便工作中使用。

<!--more-->

## 基本概念



### 矩阵的类型

#### 对角矩阵

一种特殊的方阵，其中非对角线元素全部为零。对角矩阵的转置等于自身。

#### 正交矩阵

**定义**

一个$n \times n$的矩阵$Q$被称为正交矩阵，如果它的转置等于它的逆,即:
$$
Q^TQ = QQ^T = I
$$
其中$Q^T$是$Q$的转置矩阵，$I$是$n \times n$的单位矩阵。

换句话说，正交矩阵满足$Q^T = Q^{-1}$

**正交矩阵的性质**

1. **保持长度**: 正交矩阵$Q$变换向量时保持向量的长度(或范数)。对于任意向量: $||Qx|| = ||x||$
2. **保持内积**: 正交矩阵$Q$变换向量时保持向量的内积。对于任意向量$x$和$y$: $(Qx)·(Qy) = x·y$
3. **列向量正交**: 正交矩阵的列向量是两两正交的($r_1 · r_2 = 0$)，并且每个列向量的长度为1(单位向量)。换句话说，正交矩阵的列向量组成了一个正交归一基。
4. **行向量正交**：正交矩阵的行向量是两两正交的($r_1 · r_2 = 0$)，并且每个行向量的长度为1(单位向量)。
5. **行列式**: 正交矩阵的行列式的绝对值为1，即$|det(Q)| = 1$



### 矩阵的计算

#### 矩阵的线性运算

**矩阵的加减**

矩阵的加减可正常计算，可以使用交换律

$$A + B = B + A$$

$$A - B = A + (- B)$$

**数与矩阵相乘**

满足交换律、结合律、分配律

$$(\lambda \mu) A = \lambda (\mu A)$$

$$(\lambda + \mu)A = \lambda A + \mu A$$

$$\mu (A + B) = \mu A + \mu B)$$

#### 矩阵与矩阵相乘

矩阵乘法不满足交换律，但可以使用结合律和分配律

$$(AB)C = A(BC)$$

$$\mu (AB) = (\mu A)B = A(\mu B)$$

$$ A(B + C) = AB + AC$$

#### 矩阵的转置

$$(A^T)^T = A$$

$$(A + B)^T = A^T + B^T$$

$$(\lambda A)^T = \lambda A^T$$

$$(AB)^T = B^TA^T$$

#### 逆的运算

对于多个矩阵的乘积的逆，逆运算的顺序需要反过来。比如给定三个可逆矩阵$A$，$B$和$C$，有以下公式:
$$
(ABC)^{-1} = C^{-1}B^{-1}A^{-1}
$$


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

在$Av = \lambda v$中，$Av$表示矩阵$A$作用于向量$v$，**结果是向量$v$仅仅被拉伸或缩短，而没有改变方向。**这个拉伸或缩短的比例因子就是特征值$\lambda$.



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



#### 特征值分解 (Eigenvalue Decomposition EVD)

特征值分解是一种矩阵分解技术，用于**将矩阵分解为其特征值和特征向量的组合**。

**特征值分解的定义**

对于一个$n\times n$的方阵$A$，如果$A$可以分解为:
$$
A = PDP^{-1}
$$
其中

- $P$是由矩阵$A$的特征向量按列组成的矩阵。
- $D$是一个对角矩阵，其对角元素是矩阵$A$的特征值。

那么，这个分解就是矩阵$A$的特征值分解。

注意，并不是所有方阵如果有特征值和特征向量的话，都可以进行特征值分解。判断一个方阵是否可以进行特征值分解的关键问题在于其特征向量(如果有)是否形成一个基(即线性无关)，以下是几种情况:

- **具有唯一特征值的矩阵**: 如果一个矩阵的特征值是唯一的，那么该矩阵可能没有足够的线性无关特征向量。比如某些不可对角化的方阵，即使有特征值和特征向量，仍然不可对角化。
- **重复特征值**: 对于具有重复特征值的矩阵，如果对应的特征向量的数量不足，那么矩阵也是不可对角化的。

一言以蔽之，具有足够多的线性无关特征向量的矩阵(方阵)是可对角化的，可以进行特征值分解。



然而，要进行特征值分解，矩阵$A$必须为方阵。如果$A$不是方阵，即行和列不相同时，还可以对矩阵进行特征值分解吗？此时需要使用奇异值分解(SVD)来解决(后面会提到)。

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



### 奇异值分解（Singular Value Decomposition, SVD）

https://www.cnblogs.com/sun-a/p/13543735.html

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

满足下面的内容:
$$
UU^T = I \\
VV^T = I \\
\Sigma = diag(\sigma_1, \sigma_2, ... \sigma_n) \\
\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq ... \geq \sigma_n \geq 0 \\
p = min(m, n)
$$
$U\Sigma V^T$称为矩阵$A$的奇异值分解，$\sigma$称为矩阵$A$的奇异值，$U$的列向量称为左奇异向量，$V$的列向量称为右奇异向量。

**奇异值分解的计算步骤**

1. 计算$A^TA$和$AA^T$: 找到矩阵$A$的转置并计算这两个矩阵的乘积。
2. 求特征值和特征向量: 分别计算$A^TA$和$AA^T$的特征值和特征向量。
3. 构造$U$和$V$: 使用$AA^T$的特征向量构造$U$，使用$A^TA$的特征向量构造$V$。
4. 构造$\Sigma$: 奇异值矩阵$\Sigma$的对角元素是$A^TA$或$AA^T$的非负特征值的平方根。

**奇异值分解的性质**

1. 设矩阵$A$的奇异值分解为$A = U\Sigma V^T$，以下关系成立

   1. $A^TA = (U\Sigma V^T)^T(U\Sigma V^T) = V(\Sigma^T \Sigma)V^T$
   2. $AA^T = (U\Sigma V^T)(U\Sigma V^T)^T = U(\Sigma \Sigma ^T)U^T$

   即，矩阵$A^TA$和$AA^T$的特征分解存在，且可以由矩阵$A$的奇异值分解的矩阵表示。$V$的列向量是$A^TA$的特征向量，$U$的列向量是$AA^T$的特征向量，$\Sigma$的奇异值是$A^TA$和$AA^T$的特征值的平方根。

2. 在矩阵$A$的奇异值分解中，奇异值、左奇异向量和右奇异向量之间存在下面的关系:
   $$
   AV = U\Sigma \\
   Av_j = \sigma_j u_j, j = 1, 2, ..., n
   $$
   类似的，奇异值、右奇异向量和和左奇异向量之间存在下面的关系:
   $$
   A^TU = V\Sigma^T \\
   A^Tu_j = \sigma_j v_j, j = 1, 2, ..., n \\
   A^Tu_j = 0, j = n + 1, n + 2, ..., m \\
   $$

3. 矩阵$A$的奇异值分解中，奇异值$\sigma_1$，$\sigma_2$，... $\sigma_n$是唯一的，而$U$和$V$不是唯一的。

4. 矩阵$A$和$\Sigma$的秩相等，等于正奇异值$\sigma_i$的个数r(包含重复的奇异值)(即奇异值的数量等于矩阵$A$的秩)。



