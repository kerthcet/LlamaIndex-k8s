---
title: 名字空间（Namespace）
id: namespace
date: 2018-04-12
full_link: /zh-cn/docs/concepts/overview/working-with-objects/namespaces/
short_description: >
  名字空间是 Kubernetes 用来支持隔离单个集群中的资源组的一种抽象。

aka: 
tags:
- fundamental
---



名字空间是 Kubernetes 用来支持隔离单个 {{< glossary_tooltip text="集群" term_id="cluster" >}}中的资源组的一种抽象。



名字空间用来组织集群中对象，并为集群资源划分提供了一种方法。
同一名字空间内的资源名称必须唯一，但跨名字空间时不作要求。
基于名字空间的作用域限定仅适用于名字空间作用域的对象（例如 Deployment、Services 等），
而不适用于集群作用域的对象（例如 StorageClass、Node、PersistentVolume 等）。
在一些文档里名字空间也称为命名空间。
