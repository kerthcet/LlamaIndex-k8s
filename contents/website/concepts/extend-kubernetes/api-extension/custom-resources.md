---
title: 定制资源
content_type: concept
weight: 10
---


**定制资源（Custom Resource）** 是对 Kubernetes API 的扩展。
本页讨论何时向 Kubernetes 集群添加定制资源，何时使用独立的服务。
本页描述添加定制资源的两种方法以及怎样在二者之间做出抉择。


## 定制资源  {#custom-resources}

**资源（Resource）** 是
[Kubernetes API](/zh-cn/docs/concepts/overview/kubernetes-api/) 中的一个端点，
其中存储的是某个类别的
{{< glossary_tooltip text="API 对象" term_id="object" >}}的一个集合。
例如内置的 **Pod** 资源包含一组 Pod 对象。

**定制资源（Custom Resource）** 是对 Kubernetes API 的扩展，不一定在默认的
Kubernetes 安装中就可用。定制资源所代表的是对特定 Kubernetes 安装的一种定制。
不过，很多 Kubernetes 核心功能现在都用定制资源来实现，这使得 Kubernetes 更加模块化。

定制资源可以通过动态注册的方式在运行中的集群内或出现或消失，集群管理员可以独立于集群更新定制资源。
一旦某定制资源被安装，用户可以使用 {{< glossary_tooltip text="kubectl" term_id="kubectl" >}}
来创建和访问其中的对象，就像他们为 **Pod** 这种内置资源所做的一样。

## 定制控制器   {#custom-controllers}

就定制资源本身而言，它只能用来存取结构化的数据。
当你将定制资源与**定制控制器（Custom Controller）** 结合时，
定制资源就能够提供真正的**声明式 API（Declarative API）**。

Kubernetes [声明式 API](/zh-cn/docs/concepts/overview/kubernetes-api/) 强制对职权做了一次分离操作。
你声明所用资源的期望状态，而 Kubernetes 控制器使 Kubernetes 对象的当前状态与你所声明的期望状态保持同步。
声明式 API 的这种机制与命令式 API（你**指示**服务器要做什么，服务器就去做什么）形成鲜明对比。

你可以在一个运行中的集群上部署和更新定制控制器，这类操作与集群的生命周期无关。
定制控制器可以用于任何类别的资源，不过它们与定制资源结合起来时最为有效。
[Operator 模式](/zh-cn/docs/concepts/extend-kubernetes/operator/)就是将定制资源与定制控制器相结合的。
你可以使用定制控制器来将特定于某应用的领域知识组织起来，以编码的形式构造对 Kubernetes API 的扩展。

## 我是否应该向我的 Kubernetes 集群添加定制资源？   {#should-i-add-a-cr-to-my-k8s-cluster}

在创建新的 API 时，
请考虑是[将你的 API 与 Kubernetes 集群 API 聚合起来](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)，
还是让你的 API 独立运行。

| 考虑 API 聚合的情况 | 优选独立 API 的情况 |
| ---------------------------- | ---------------------------- |
| 你的 API 是[声明式的](#declarative-apis)。 | 你的 API 不符合[声明式](#declarative-apis)模型。 |
| 你希望可以是使用 `kubectl` 来读写你的新资源类别。 | 不要求 `kubectl` 支持。 |
| 你希望在 Kubernetes UI （如仪表板）中和其他内置类别一起查看你的新资源类别。 | 不需要 Kubernetes UI 支持。 |
| 你在开发新的 API。 | 你已经有一个提供 API 服务的程序并且工作良好。 |
| 你有意愿取接受 Kubernetes 对 REST 资源路径所作的格式限制，例如 API 组和名字空间。（参阅 [API 概述](/zh-cn/docs/concepts/overview/kubernetes-api/)） | 你需要使用一些特殊的 REST 路径以便与已经定义的 REST API 保持兼容。 |
| 你的资源可以自然地界定为集群作用域或集群中某个名字空间作用域。 | 集群作用域或名字空间作用域这种二分法很不合适；你需要对资源路径的细节进行控制。 |
| 你希望复用 [Kubernetes API 支持特性](#common-features)。  | 你不需要这类特性。 |

### 声明式 API   {#declarative-apis}

典型地，在声明式 API 中：

- 你的 API 包含相对而言为数不多的、尺寸较小的对象（资源）。
- 对象定义了应用或者基础设施的配置信息。
- 对象更新操作频率较低。
- 通常需要人来读取或写入对象。
- 对象的主要操作是 CRUD 风格的（创建、读取、更新和删除）。
- 不需要跨对象的事务支持：API 对象代表的是期望状态而非确切实际状态。

命令式 API（Imperative API）与声明式有所不同。
以下迹象表明你的 API 可能不是声明式的：

- 客户端发出“做这个操作”的指令，之后在该操作结束时获得同步响应。
- 客户端发出“做这个操作”的指令，并获得一个操作 ID，之后需要检查一个 Operation（操作）
  对象来判断请求是否成功完成。
- 你会将你的 API 类比为远程过程调用（Remote Procedure Call，RPC）。
- 直接存储大量数据；例如每个对象几 kB，或者存储上千个对象。
- 需要较高的访问带宽（长期保持每秒数十个请求）。
- 存储有应用来处理的最终用户数据（如图片、个人标识信息（PII）等）或者其他大规模数据。
- 在对象上执行的常规操作并非 CRUD 风格。
- API 不太容易用对象来建模。
- 你决定使用操作 ID 或者操作对象来表现悬决的操作。

## 我应该使用一个 ConfigMap 还是一个定制资源？   {#should-i-use-a-configmap-or-a-cr}

如果满足以下条件之一，应该使用 ConfigMap：

* 存在一个已有的、文档完备的配置文件格式约定，例如 `mysql.cnf` 或 `pom.xml`。
* 你希望将整个配置文件放到某 configMap 中的一个主键下面。
* 配置文件的主要用途是针对运行在集群中 Pod 内的程序，供后者依据文件数据配置自身行为。
* 文件的使用者期望以 Pod 内文件或者 Pod 内环境变量的形式来使用文件数据，
  而不是通过 Kubernetes API。
* 你希望当文件被更新时通过类似 Deployment 之类的资源完成滚动更新操作。

{{< note >}}
请使用 {{< glossary_tooltip text="Secret" term_id="secret" >}} 来保存敏感数据。
Secret 类似于 configMap，但更为安全。
{{< /note >}}

如果以下条件中大多数都被满足，你应该使用定制资源（CRD 或者 聚合 API）：

* 你希望使用 Kubernetes 客户端库和 CLI 来创建和更改新的资源。
* 你希望 `kubectl` 能够直接支持你的资源；例如，`kubectl get my-object object-name`。
* 你希望构造新的自动化机制，监测新对象上的更新事件，并对其他对象执行 CRUD
  操作，或者监测后者更新前者。
* 你希望编写自动化组件来处理对对象的更新。
* 你希望使用 Kubernetes API 对诸如 `.spec`、`.status` 和 `.metadata` 等字段的约定。
* 你希望对象是对一组受控资源的抽象，或者对其他资源的归纳提炼。

## 添加定制资源   {#adding-custom-resources}

Kubernetes 提供了两种方式供你向集群中添加定制资源：

- CRD 相对简单，创建 CRD 可以不必编程。
- [API 聚合](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)需要编程，
  但支持对 API 行为进行更多的控制，例如数据如何存储以及在不同 API 版本间如何转换等。

Kubernetes 提供这两种选项以满足不同用户的需求，这样就既不会牺牲易用性也不会牺牲灵活性。

聚合 API 指的是一些下位的 API 服务器，运行在主 API 服务器后面；主 API
服务器以代理的方式工作。这种组织形式称作
[API 聚合（API Aggregation，AA）](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/) 。
对用户而言，看起来仅仅是 Kubernetes API 被扩展了。

CRD 允许用户创建新的资源类别同时又不必添加新的 API 服务器。
使用 CRD 时，你并不需要理解 API 聚合。

无论以哪种方式安装定制资源，新的资源都会被当做定制资源，以便与内置的
Kubernetes 资源（如 Pods）相区分。

{{< note >}}
避免将定制资源用于存储应用、最终用户或监控数据：
将应用数据存储在 Kubernetes API 内的架构设计通常代表一种过于紧密耦合的设计。

在架构上，[云原生](https://www.cncf.io/about/faq/#what-is-cloud-native)应用架构倾向于各组件之间的松散耦合。
如果部分工作负载需要支持服务来维持其日常运转，则这种支持服务应作为一个组件运行或作为一个外部服务来使用。
这样，工作负载的正常运转就不会依赖 Kubernetes API 了。
{{< /note >}}

## CustomResourceDefinitions

[CustomResourceDefinition](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
API 资源允许你定义定制资源。
定义 CRD 对象的操作会使用你所设定的名字和模式定义（Schema）创建一个新的定制资源，
Kubernetes API 负责为你的定制资源提供存储和访问服务。
CRD 对象的名称必须是合法的
[DNS 子域名](/zh-cn/docs/concepts/overview/working-with-objects/names#dns-subdomain-names)。

CRD 使得你不必编写自己的 API 服务器来处理定制资源，不过其背后实现的通用性也意味着你所获得的灵活性要比
[API 服务器聚合](#api-server-aggregation)少很多。

关于如何注册新的定制资源、使用新资源类别的实例以及如何使用控制器来处理事件，
相关的例子可参见[定制控制器示例](https://github.com/kubernetes/sample-controller)。

## API 服务器聚合  {#api-server-aggregation}

通常，Kubernetes API 中的每个资源都需要处理 REST 请求和管理对象持久性存储的代码。
Kubernetes API 主服务器能够处理诸如 **Pod** 和 **Service** 这些内置资源，
也可以按通用的方式通过 [CRD](#customresourcedefinitions) 来处理定制资源。

[聚合层（Aggregation Layer）](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)
使得你可以通过编写和部署你自己的 API 服务器来为定制资源提供特殊的实现。
主 API 服务器将针对你要处理的定制资源的请求全部委托给你自己的 API 服务器来处理，
同时将这些资源提供给其所有客户端。

## 选择添加定制资源的方法   {#choosing-a-method-for-adding-cr}

CRD 更为易用；聚合 API 则更为灵活。请选择最符合你的需要的方法。

通常，如何存在以下情况，CRD 可能更合适：

* 定制资源的字段不多；
* 你在组织内部使用该资源或者在一个小规模的开源项目中使用该资源，而不是在商业产品中使用。

### 比较易用性  {#compare-ease-of-use}

CRD 比聚合 API 更容易创建。

| CRD                        | 聚合 API       |
| --------------------------- | -------------- |
| 无需编程。用户可选择任何语言来实现 CRD 控制器。 | 需要编程，并构建可执行文件和镜像。 |
| 无需额外运行服务；CRD 由 API 服务器处理。 | 需要额外创建服务，且该服务可能失效。 |
| 一旦 CRD 被创建，不需要持续提供支持。Kubernetes 主控节点升级过程中自动会带入缺陷修复。 | 可能需要周期性地从上游提取缺陷修复并更新聚合 API 服务器。 |
| 无需处理 API 的多个版本；例如，当你控制资源的客户端时，你可以更新它使之与 API 同步。 | 你需要处理 API 的多个版本；例如，在开发打算与很多人共享的扩展时。 |

### 高级特性与灵活性  {#advanced-features-and-flexibility}

聚合 API 可提供更多的高级 API 特性，也可对其他特性实行定制；例如，对存储层进行定制。

| 特性    | 描述        | CRD | 聚合 API       |
| ------- | ----------- | ---- | -------------- |
| 合法性检查 | 帮助用户避免错误，允许你独立于客户端版本演化 API。这些特性对于由很多无法同时更新的客户端的场合。| 可以。大多数验证可以使用 [OpenAPI v3.0 合法性检查](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#validation) 来设定。其他合法性检查操作可以通过添加[合法性检查 Webhook](/zh-cn/docs/reference/access-authn-authz/admission-controllers/#validatingadmissionwebhook-alpha-in-1-8-beta-in-1-9)来实现。 | 可以，可执行任何合法性检查。|
| 默认值设置 | 同上 | 可以。可通过 [OpenAPI v3.0 合法性检查](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#defaulting)的 `default` 关键词（自 1.17 正式发布）或[更改性（Mutating）Webhook](/zh-cn/docs/reference/access-authn-authz/admission-controllers/#mutatingadmissionwebhook)来实现（不过从 etcd 中读取老的对象时不会执行这些 Webhook）。 | 可以。 |
| 多版本支持 | 允许通过两个 API 版本同时提供同一对象。可帮助简化类似字段更名这类 API 操作。如果你能控制客户端版本，这一特性将不再重要。 | [可以](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definition-versioning)。 | 可以。 |
| 定制存储 | 支持使用具有不同性能模式的存储（例如，要使用时间序列数据库而不是键值存储），或者因安全性原因对存储进行隔离（例如对敏感信息执行加密）。 | 不可以。 | 可以。 |
| 定制业务逻辑 | 在创建、读取、更新或删除对象时，执行任意的检查或操作。 | 可以。要使用 [Webhook](/zh-cn/docs/reference/access-authn-authz/extensible-admission-controllers/#admission-webhooks)。 | 可以。 |
| 支持 scale 子资源 | 允许 HorizontalPodAutoscaler 和 PodDisruptionBudget 这类子系统与你的新资源交互。 | [可以](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#scale-subresource)。 | 可以。 |
| 支持 status 子资源 | 允许在用户写入 spec 部分而控制器写入 status 部分时执行细粒度的访问控制。允许在对定制资源的数据进行更改时增加对象的代际（Generation）；这需要资源对 spec 和 status 部分有明确划分。| [可以](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#status-subresource)。 | 可以。 |
| 其他子资源 | 添加 CRUD 之外的操作，例如 "logs" 或 "exec"。 | 不可以。 | 可以。 |
| strategic-merge-patch | 新的端点要支持标记了 `Content-Type: application/strategic-merge-patch+json` 的 PATCH 操作。对于更新既可在本地更改也可在服务器端更改的对象而言是有用的。要了解更多信息，可参见[使用 `kubectl patch` 来更新 API 对象](/zh-cn/docs/tasks/manage-kubernetes-objects/update-api-object-kubectl-patch/)。 | 不可以。 | 可以。 |
| 支持协议缓冲区 | 新的资源要支持想要使用协议缓冲区（Protocol Buffer）的客户端。 | 不可以。 | 可以。 |
| OpenAPI Schema | 是否存在新资源类别的 OpenAPI（Swagger）Schema 可供动态从服务器上读取？是否存在机制确保只能设置被允许的字段以避免用户犯字段拼写错误？是否实施了字段类型检查（换言之，不允许在 `string` 字段设置 `int` 值）？ | 可以，依据 [OpenAPI v3.0 合法性检查](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#validation) 模式（1.16 中进入正式发布状态）。 | 可以。|

### 公共特性  {#common-features}

与在 Kubernetes 平台之外实现定制资源相比，
无论是通过 CRD 还是通过聚合 API 来创建定制资源，你都会获得很多 API 特性：

| 功能特性 | 具体含义     |
| -------- | ------------ |
| CRUD | 新的端点支持通过 HTTP 和 `kubectl` 发起的 CRUD 基本操作 |
| 监测（Watch） | 新的端点支持通过 HTTP 发起的 Kubernetes Watch 操作 |
| 发现（Discovery） | 类似 `kubectl` 和仪表盘（Dashboard）这类客户端能够自动提供列举、显示、在字段级编辑你的资源的操作 |
| json-patch | 新的端点支持带 `Content-Type: application/json-patch+json` 的 PATCH 操作 |
| merge-patch | 新的端点支持带 `Content-Type: application/merge-patch+json` 的 PATCH 操作 |
| HTTPS | 新的端点使用 HTTPS |
| 内置身份认证 | 对扩展的访问会使用核心 API 服务器（聚合层）来执行身份认证操作 |
| 内置鉴权授权 | 对扩展的访问可以复用核心 API 服务器所使用的鉴权授权机制；例如，RBAC |
| Finalizers | 在外部清除工作结束之前阻止扩展资源被删除 |
| 准入 Webhooks | 在创建、更新和删除操作中对扩展资源设置默认值和执行合法性检查 |
| UI/CLI 展示 | `kubectl` 和仪表盘（Dashboard）可以显示扩展资源 |
| 区分未设置值和空值 | 客户端能够区分哪些字段是未设置的，哪些字段的值是被显式设置为零值的  |
| 生成客户端库 | Kubernetes 提供通用的客户端库，以及用来生成特定类别客户端库的工具 |
| 标签和注解 | 提供涵盖所有对象的公共元数据结构，且工具知晓如何编辑核心资源和定制资源的这些元数据 |

## 准备安装定制资源   {#preparing-to-install-a-cr}

在向你的集群添加定制资源之前，有些事情需要搞清楚。

### 第三方代码和新的失效点的问题   {#third-party-code-and-new-points-of-failure}

尽管添加新的 CRD 不会自动带来新的失效点（Point of
Failure），例如导致第三方代码被在 API 服务器上运行，
类似 Helm Charts 这种软件包或者其他安装包通常在提供 CRD
的同时还包含带有第三方代码的 Deployment，负责实现新的定制资源的业务逻辑。

安装聚合 API 服务器时，也总会牵涉到运行一个新的 Deployment。

### 存储    {#storage}

定制资源和 ConfigMap 一样也会消耗存储空间。创建过多的定制资源可能会导致
API 服务器上的存储空间超载。

聚合 API 服务器可以使用主 API 服务器相同的存储。如果是这样，你也要注意此警告。

### 身份认证、鉴权授权以及审计    {#authentication-authorization-and-auditing}

CRD 通常与 API 服务器上的内置资源一样使用相同的身份认证、鉴权授权和审计日志机制。

如果你使用 RBAC 来执行鉴权授权，大多数 RBAC 角色都不会授权对新资源的访问
（除了 cluster-admin 角色以及使用通配符规则创建的其他角色）。
你要显式地为新资源的访问授权。CRD 和聚合 API 通常在交付时会包含针对所添加的类别的新的角色定义。

聚合 API 服务器可能会使用主 API 服务器相同的身份认证、鉴权授权和审计机制，也可能不会。

## 访问定制资源   {#accessing-a-custom-resources}

Kubernetes [客户端库](/zh-cn/docs/reference/using-api/client-libraries/)可用来访问定制资源。
并非所有客户端库都支持定制资源。**Go** 和 **Python** 客户端库是支持的。

当你添加了新的定制资源后，可以用如下方式之一访问它们：

- `kubectl`
- Kubernetes 动态客户端
- 你所编写的 REST 客户端
- 使用 [Kubernetes 客户端生成工具](https://github.com/kubernetes/code-generator)所生成的客户端。
  生成客户端的工作有些难度，不过某些项目可能会随着 CRD 或聚合 API 一起提供一个客户端。

## {{% heading "whatsnext" %}}

* 了解如何[使用聚合层扩展 Kubernetes API](/zh-cn/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)
* 了解如何[使用 CustomResourceDefinition 来扩展 Kubernetes API](/zh-cn/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
