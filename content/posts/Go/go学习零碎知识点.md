---
# type: docs 
title: Go学习零碎知识点
date: 2024-06-13T09:52:19+08:00
featured: false
draft: true
comment: true
toc: true
reward: true
pinned: false
carousel: false
series:
categories: ["Note"]
tags: ["Go"]
images: []
---

在学习新的语言当中肯定会出现很多问题，而系统性的记录知识点以及用法过于浪费时间。所以我会记录一些在学习go语言当中遇到的一些零碎的知识点，用于加深对go语言的理解。

<!--more-->

go语言学习网站 (附带IDE): https://tour.go-zh.org/list

## 编译与运行问题

在go语言中，统一路径的文件被认为是在同一个package里面，在相同路径下，里面的文件必须设置为相同的package名。比如同一目录的文件设置为`package main`，如果有文件设置为其他package名称(比如`package foo`)，则会在编译时报错。

对于编译问题，默认情况下使用`go build`，而`go build`会构建整个包，比如在当前路径下使用`go build -o main`会将当前路径下的所有go文件执行编译。我们也可以使用`go build -o main main.go`来编译一个文件，但是不建议这样使用，因为 go强调包的整体构建。

如果只想要临时构建和运行一个文件，可以使用`go run`，比如`go run main.go`。



## go语言代码细节

### 声明与赋值规范

常用声明规范为: `keyword name type`。比如`var a int`，代表的是关键字是variable，变量名是a，是一个整数类型。

常用赋值规范可以用=或:=

`=`用于变量的赋值和重新赋值，前提是变量已经声明过。

```go
var a int = 1

//也可以
var a = 1

//也可以
var a int
a = 1
```

`:=`用于短变量的声明赋值，可以视为声明+赋值，这种方式不用声明keyword

```go
a := 1
```



### 语句功能

##### defer

defer语句会将函数推迟到外层函数返回之后执行。推迟调用的函数其参数会立即求值，但直到外层函数返回前该函数都不会被调用。

defer的逻辑是将defer的信息压入到一个栈中，当外层函数返回后，被推迟的信息会按照栈的后进先出的方式进行调用。

