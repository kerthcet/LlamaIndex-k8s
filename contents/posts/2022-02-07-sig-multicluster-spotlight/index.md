---
layout: blog
title: "关注 SIG Multicluster"
date: 2022-02-07
slug: sig-multicluster-spotlight-2022
---

**作者：** Dewan Ahmed (Aiven) 和 Chris Short (AWS)

## 简介

[SIG Multicluster](https://github.com/kubernetes/community/tree/master/sig-multicluster)
是专注于如何拓展 Kubernetes 的概念并将其用于集群边界之外的 SIG。
以往 Kubernetes 资源仅在 Kubernetes Resource Universe (KRU) 这个边界内进行交互，其中 KRU 不是一个实际的 Kubernetes 概念。
即使是现在，Kubernetes 集群对自身或其他集群并不真正了解。集群标识符的缺失就是一个例子。
随着多云和多集群部署日益普及，SIG Multicluster 所做的工作越来越受到关注。
在这篇博客中，[来自 Google 的 Jeremy Olmsted-Thompson](https://twitter.com/jeremyot) 和
[来自 AWS 的 Chris Short](https://twitter.com/ChrisShort) 讨论了 SIG Multicluster
正在解决的一些有趣的问题和以及大家如何参与其中。
为简洁起见，下文将使用他们两位的首字母 **JOT** 和 **CS**。

## 谈话总结

**CS**：SIG Multicluster 存在多久了？SIG 在起步阶段情况如何？你参与这个 SIG 多长时间了？

**JOT**：我在 SIG Multicluster 工作了将近两年。我所知道的关于初创时期的情况都来自传说，但即使在早期，也一直是为了解决相同的问题。
早期工作的例子之一是 [KubeFed](https://github.com/kubernetes-sigs/kubefed)。
我认为仍然有一些人在使用 KubeFed，但它只是一小部分。
那时，我认为人们在部署大量 Kubernetes 集群时，还没有达到我们拥有大量实际具体用例的地步。
像 KubeFed 和 [Cluster Registry](https://github.com/kubernetes-retired/cluster-registry)
这样的项目就是在那个时候开发的，当时的需求可以与这些项目相关联。
这些项目的动机是如何解决我们认为在开始扩展到多个集群时 **会遇到的问题**。
老实说，在某些方面，当时它试图做得太多了。

**CS**：KubeFed 与 SIG Multicluster 的现状有何不同？**初创期** 与 **现在** 有何不同？

**JOT**：嗯嗯，这就像总是要预防潜在问题而不是解决眼前具体的问题。我认为在 2019 年底，
SIG Multicluster 工作有所放缓，我们通过最近最活跃的项目之一
[SIG Multicluster services (MCS)](https://github.com/kubernetes-sigs/mcs-api) 将其重新拾起。

现在我们向解决实际的具体问题开始转变。比如说。

> 我的工作负载分布在多个集群中，我需要它们相互通信。

好吧，这是非常直接的，我们也知道需要解决这个问题。
首先，让我们确保这些项目可以在一个通用的 API 上协同工作，这样你就可以获得与 Kubernetes 相同的可移植性。

目前有一些 MCS API 的实现，并且更多的实现正在开发中。但是，我们没有建立一个实现，
因为取决于你的部署方式不同，可能会有数百种实现。
只要你所需要的基本的多集群服务功能，它就可以在你想要的任何背景下工作，无论是 Submariner、GKE 还是服务网格。

我最喜欢的“过去与现在“的例子是集群 ID。几年前曾经有过定义集群 ID 的尝试。
针对这个概念，有很多非常好的想法。例如，我们如何使集群 ID 在多个集群中是唯一的。
我们如何使这个 ID 全球范围内唯一，以便它在各个通讯中发挥作用？
假设有团队被收购或合并 - 集群 ID 对于这些团队仍然是唯一的吗？

在 Multicluster 服务的相关工作中，我们发现需要一个实际的集群 ID，并且这一需求非常具体。
为了满足这一特定需求，我们不再考虑一个个 Kubernetes 集群，而是考虑 ClusterSets — 在某种范围内协同工作的集群分组。
与考虑所有时间点和所有空间位置上存在的集群相比，这一范畴要窄得多。
这一概念还让实现者具备了定义边界（ClusterSet）的灵活性，在该边界之外，该集群 ID 将不再是唯一的。

**CS**：你对 SIG Multicluster 的现状有何看法，你希望未来达到什么样的目标？

**JOT**：有一些项目正在起步，例如 Work API。 在未来，我认为围绕着如何跨集群部署应用的一些共同做法将会发展起来。
> 如果我的集群部署在不同的地区，那么最好的方式是什么？

答案几乎总是“视情况而定”。你为什么要这样做？是因为某种合规性使你关注位置吗？是性能问题吗？是可用性吗？

我认为，在我们有了集群 ID 之后，重新审视注册表模式可能是很自然的一步，也就是说，
你如何将这些集群真正关联在一起？也许你有一个分布式部署，你在世界各地的数据中心运行。
我想随着多集群特性的进一步开发，扩展该领域的 API 将变得很重要。
这实际上取决于社区开始使用这些工具做什么。

**CS**：在 Kubernetes 的早期，我们只有寥寥几个大型的 Kubernetes 集群，而现在我们面对的是大量的小型 Kubernetes 集群，就像我自己所在的开发环境中就使用了多个集群。
这种从几个大集群到许多小集群的转变对 SIG 有何影响？它是否加快了工作进度或在某种程度上使得问题变得更困难？

**JOT**：我认为它带来了很多需要解决的歧义。最初，你可能拥有一个 dev 集群、一个 staging 集群和一个 prod 集群。
当引入了多区域的考量时，我们开始在每个区域部署 dev/staging/prod 集群。
再后来，有时由于合规性或某些法规问题，集群确实需要更多的隔离。
因此，我们最终会有很多集群。我认为在你究竟应该有多少个集群上找到平衡是很重要的。Kubernetes 的强大之处在于能够部署由单个控制平面管理的大量事物。
因此，并不是每个被部署的工作负载都应该在自己的集群中。
但是，我认为同样很明显的是，我们不能将所有工作负载都放在一个集群中。

**CS**：你最喜欢 SIG 的哪些方面？

**JOT**：问题的复杂性、人的因素和领域的新颖性。我们还没有正确的答案，我们必须找到正确的答案。
一开始，我们甚至无法考虑多集群，因为无法跨集群连接服务。
现在我们开始着手解决这些问题，我认为这是一个非常有趣的地方，因为我预计 SIG 在未来几年会变得更加繁忙。
这是一个协作很密切的团体，我们绝对希望更多的人参与、加入我们，提出他们的问题和想法。

**CS**：你认为是什么让人们留在这个群体中？疫情对你有何影响？

**JOT**：我认为在疫情期间这个群体肯定会变得安静一些。但在大多数情况下，这是一个非常分散的小组，
因此无论你在会议室或者在家中参加我们的每周会议，都不会产生太大的影响。在疫情期间，很多人有时间专注于他们接下来的规模和增长。
我认为这就是让人们留在团队中的原因 - 我们有真正的问题需要解决，这些问题在这个领域是非常新颖的、有趣的。

## 结束语

**CS**：这就是我们今天的全部内容，感谢 Jeremy 的时间。

**JOT**：谢谢 Chris。我们的[双周会议](https://github.com/kubernetes/community/tree/master/sig-multicluster#meetings)
欢迎所有人参加。我们希望尽可能多的人前来，并欢迎所有问题与想法。
这是一个新的领域，如果能让社区发展起来，那就太好了。
