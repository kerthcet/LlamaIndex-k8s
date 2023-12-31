---
title: " OpenStack 上的 Kubernetes "
date: 2015-05-19
slug: kubernetes-on-openstack
---

[![](https://3.bp.blogspot.com/-EOrCHChZJZE/VVZzq43g6CI/AAAAAAAAF-E/JUilRHk369E/s400/Untitled%2Bdrawing.jpg)](https://3.bp.blogspot.com/-EOrCHChZJZE/VVZzq43g6CI/AAAAAAAAF-E/JUilRHk369E/s1600/Untitled%2Bdrawing.jpg)


今天，[OpenStack 基金会](https://www.openstack.org/foundation/)通过在其[社区应用程序目录](http://apps.openstack.org/)中包含 Kubernetes，使您更容易在 OpenStack 云上部署和管理 Docker 容器集群。
今天在温哥华 OpenStack 峰会上的主题演讲中，OpenStack 基金会的首席运营官：Mark Collier 和 [Mirantis](https://www.mirantis.com/) 产品线经理 Craig Peters 通过利用 OpenStack 云中已经存在的计算、存储、网络和标识系统，在几秒钟内启动了 Kubernetes 集群，展示了社区应用程序目录的工作流。

目录中的条目不仅包括[启动 Kubernetes 集群](http://apps.openstack.org/#tab=murano-apps&asset=Kubernetes%20Cluster)的功能，还包括部署在 Kubernetes 管理的 Docker 容器中的一系列应用程序。这些应用包括：



-
Apache web 服务器
-
Nginx web 服务器
-
Crate - Docker的分布式数据库
-
GlassFish - Java EE 7 应用服务器
-
Tomcat - 一个开源的 web 服务器和 servlet 容器
-
InfluxDB - 一个开源的、分布式的、时间序列数据库
-
Grafana -   InfluxDB 的度量仪表板
-
Jenkins - 一个可扩展的开放源码持续集成服务器
-
MariaDB 数据库
-
MySql 数据库
-
Redis - 键-值缓存和存储
-
PostgreSQL 数据库
-
MongoDB NoSQL 数据库
-
Zend 服务器 - 完整的 PHP 应用程序平台

此列表将会增长，并在[此处](https://opendev.org/x/k8s-docker-suite-app-murano/src/branch/master/Kubernetes)进行策划。您可以检查（并参与）YAML 文件，该文件告诉 Murano 如何根据[此处](https://opendev.org/x/k8s-docker-suite-app-murano/src/branch/master/Kubernetes/KubernetesCluster/package/Classes/KubernetesCluster.yaml)定义来安装和启动 ...apps/blob/master/Docker/Kubernetes/KubernetesCluster/package/Classes/KubernetesCluster.yaml)安装和启动 Kubernetes 集群。

[Kubernetes 开源项目](https://github.com/GoogleCloudPlatform/kubernetes)继续受到社区的欢迎，并且势头越来越好，GitHub 上有超过 11000 个提交和 7648 颗星。从 Red Hat 和 Intel 到 CoreOS 和 Box.net，它已经代表了从企业 IT 到前沿创业企业的一系列客户。我们鼓励您尝试一下，给我们您的反馈，并参与到我们不断增长的社区中来。



- Martin Buhr, Kubernetes 开源项目产品经理

