---
title: " Kubernetes Release: 0.15.0 "
date: 2015-04-16
slug: kubernetes-release-0150
---


Release 说明：


* 启用 1beta3 API 并将其设置为默认 API 版本 ([#6098][1])
* 增加了多端口服务([#6182][2])
    * 新入门指南
    * 多节点本地启动指南 ([#6505][3])
    * Google 云平台上的 Mesos ([#5442][4])
    * Ansible 安装说明 ([#6237][5])
* 添加了一个控制器框架 ([#5270][6], [#5473][7])
* Kubelet 现在监听一个安全的 HTTPS 端口 ([#6380][8])
* 使 kubectl 错误更加友好 ([#6338][9])
* apiserver 现在支持客户端 cert 身份验证 ([#6190][10])
* apiserver 现在限制了它处理的并发请求的数量 ([#6207][11])
* 添加速度限制删除 pod ([#6355][12])
* 将平衡资源分配算法作为优先级函数实现在调度程序包中 ([#6150][13])
* 从主服务器启用日志收集功能 ([#6396][14])
* 添加了一个 api 端口来从 Pod 中提取日志 ([#6497][15])
* 为调度程序添加了延迟指标 ([#6368][16])
* 为 REST 客户端添加了延迟指标 ([#6409][17])


* etcd 现在在 master 上的一个 pod 中运行 ([#6221][18])
* nginx 现在在 master上的容器中运行 ([#6334][19])
* 开始为主组件构建 Docker 镜像 ([#6326][20])
* 更新了 GCE 程序以使用 gcloud 0.9.54 ([#6270][21])
* 更新了 AWS 程序来修复区域与区域语义 ([#6011][22])
* 记录镜像 GC 失败时的事件 ([#6091][23])
* 为 kubernetes 客户端添加 QPS 限制器 ([#6203][24])
* 减少运行 make release 所需的时间 ([#6196][25])
* 新卷的支持
    * 添加 iscsi 卷插件 ([#5506][26])
    * 添加 glusterfs 卷插件 ([#6174][27])
    * AWS EBS 卷支持 ([#5138][28])
* 更新到 heapster 版本到 v0.10.0 ([#6331][29])
* 更新到 etcd 2.0.9 ([#6544][30])
* 更新到 Kibana 到 v1.2 ([#6426][31])
* 漏洞修复
    * 如果服务的公共 IP 发生变化，Kube-proxy现在会更新iptables规则 ([#6123][32])
    * 如果初始创建失败，则重试 kube-addons 创建 ([#6200][33])
    * 使 kube-proxy 对耗尽文件描述符更具弹性 ([#6727][34])

要下载，请访问 https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.15.0

[1]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6098 "在 master 中默认启用 v1beta3 api 版本"
[2]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6182 "实现多端口服务"
[3]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6505 "Docker 多节点"
[4]: https://github.com/GoogleCloudPlatform/kubernetes/pull/5442 "谷歌云平台上 Mesos 入门指南"
[5]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6237 "示例 ansible 设置仓库"
[6]: https://github.com/GoogleCloudPlatform/kubernetes/pull/5270 "控制器框架"
[7]: https://github.com/GoogleCloudPlatform/kubernetes/pull/5473 "添加 DeltaFIFO（控制器框架块）"
[8]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6380 "将 kubelet 配置为使用 HTTPS (获得 2)"
[9]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6338 "返回用于配置验证的类型化错误，并简化错误"
[10]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6190 "添加客户端证书认证"
[11]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6207 "为服务器处理的正在运行的请求数量添加一个限制。"
[12]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6355 "添加速度限制删除 pod"
[13]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6150 "将均衡资源分配算法作为优先级函数实现在调度程序包中。"
[14]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6396 "启用主服务器收集日志。"
[15]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6497 "pod 子日志资源"
[16]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6368 "将基本延迟指标添加到调度程序。"
[17]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6409 "向 REST 客户端添加延迟指标"
[18]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6221 "在 pod 中运行 etcd 2.0.5"
[19]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6334 "添加一个 nginx docker 镜像用于主程序。"
[20]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6326 "为主组件创建 Docker 镜像"
[21]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6270 "gcloud 0.9.54 的更新"

[22]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6011 "修复 AWS 区域 与 zone"
[23]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6091 "记录镜像 GC 失败时的事件。"
[24]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6203 "向 kubernetes 客户端添加 QPS 限制器。"
[25]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6196 "在 `make release` 的构建和打包阶段并行化架构"
[26]: https://github.com/GoogleCloudPlatform/kubernetes/pull/5506 "添加 iscsi 卷插件"
[27]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6174 "实现 glusterfs 卷插件"
[28]: https://github.com/GoogleCloudPlatform/kubernetes/pull/5138 "AWS EBS 卷支持"
[29]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6331 "将 heapster 版本更新到 v0.10.0"
[30]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6544 "构建 etcd 镜像(版本 2.0.9)，并将 kubernetes 集群升级到新版本"
[31]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6426 "更新 Kibana 到 v1.2，它对 Elasticsearch 的位置进行了参数化"
[32]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6123 "修复了 kube-proxy 中的一个错误，如果一个服务的公共 ip 发生变化，它不会更新 iptables 规则"
[33]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6200 "如果 kube-addons 创建失败，请重试 kube-addons 创建。"
[34]: https://github.com/GoogleCloudPlatform/kubernetes/pull/6727 "pkg/proxy: fd 用完后引起恐慌"

