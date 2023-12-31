---
title: “基于容器的应用程序设计原理”
date: 2018-03-15
slug: principles-of-container-app-design
---

如今，可以将几乎所有应用程序放入容器中并运行它。
但是，创建云原生应用程序（由 Kubernetes 等云原生平台自动有效地编排的容器化应用程序）需要付出额外的努力。
云原生应用程序会预期失败；
它们可以可靠的运行和扩展，即使基础架构出现故障。
为了提供这样的功能，像 Kubernetes 这样的云原生平台对应用程序施加了一系列约定和约束。
这些合同确保运行的应用程序符合某些约束条件，并允许平台自动执行应用程序管理。


我总结了容器化应用要成为彻底的云原生应用所要遵从的[七个原则][1]。

| ----- |
| ![][2]  |
| 容器设计原则 |


这七个原则涵盖了构建时间和运行时问题。

####  建立时间

* **单独关注点：** 每个容器都解决了一个单独的关注点，并做到了。
* **自包含：** 容器仅依赖于Linux内核的存在。 构建容器时会添加其他库。
* **镜像不可变性：** 容器化应用程序是不可变的，并且一旦构建，就不会在不同环境之间发生变化。

####  运行时

* **高度可观察性：** 每个容器都必须实现所有必要的API，以帮助平台以最佳方式观察和管理应用程序。
* **生命周期一致性：** 容器必须具有读取来自平台的事件并通过对这些事件做出反应来进行一致性的方式。
* **进程可丢弃：** 容器化的应用程序必须尽可能短暂，并随时可以被另一个容器实例替换。
* **运行时可约束** 每个容器必须声明其资源需求，并根据所标明的需求限制其资源使用。
构建时间原则可确保容器具有正确的颗粒度，一致性和适当的结构。
运行时原则规定了必须执行哪些功能才能使容器化的应用程序具有云原生功能。
遵循这些原则有助于确保您的应用程序适合Kubernetes中的自动化。

白皮书可以免费下载：

要了解有关为Kubernetes设计云原生应用程序的更多信息，请翻阅我的[Kubernetes 模式][3]这本书。

— [Bilgin Ibryam][4]，首席架构师，Red Hat

推特：   
博客： [http://www.ofbizian.com][5]  
领英：

Bilgin Ibryam（@bibryam）是 Red Hat 的首席架构师，ASF 的开源提交者，博客，作者和发言人。
他是骆驼设计模式和 Kubernetes 模式书籍的作者。
在日常工作中，Bilgin 乐于指导，培训和领导团队，以使他们在分布式系统，微服务，容器和云原生应用程序方面取得成功。


[1]: https://www.redhat.com/en/resources/cloud-native-container-design-whitepaper
[2]: https://lh5.googleusercontent.com/1XqojkVC0CET1yKCJqZ3-0VWxJ3W8Q74zPLlqnn6eHSJsjHOiBTB7EGUX5o_BOKumgfkxVdgBeLyoyMfMIXwVm9p2QXkq_RRy2mDJG1qEExJDculYL5PciYcWfPAKxF2-DGIdiLw
[3]: http://leanpub.com/k8spatterns/
[4]: http://twitter.com/bibryam
[5]: http://www.ofbizian.com/

