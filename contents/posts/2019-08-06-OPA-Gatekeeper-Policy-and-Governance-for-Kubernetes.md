---
layout: blog
title: "OPA Gatekeeper：Kubernetes 的策略和管理"
date: 2019-08-06
slug: OPA-Gatekeeper-Policy-and-Governance-for-Kubernetes 
---

**作者：** Rita Zhang (Microsoft), Max Smythe (Google), Craig Hooper (Commonwealth Bank AU), Tim Hinrichs (Styra), Lachie Evenson (Microsoft), Torin Sandall (Styra)

可以从项目 [Open Policy Agent Gatekeeper](https://github.com/open-policy-agent/gatekeeper) 中获得帮助，在 Kubernetes 环境下实施策略并加强治理。在本文中，我们将逐步介绍该项目的目标，历史和当前状态。

以下是 Kubecon EU 2019 会议的录音，帮助我们更好地开展与 Gatekeeper 合作：

* [简介：开放策略代理 Gatekeeper](https://youtu.be/Yup1FUc2Qn0)
* [深入研究：开放策略代理](https://youtu.be/n94_FNhuzy4)

## 出发点 

如果您所在的组织一直在使用 Kubernetes，您可能一直在寻找如何控制终端用户在集群上的行为，以及如何确保集群符合公司政策。这些策略可能需要满足管理和法律要求，或者符合最佳执行方法和组织惯例。使用 Kubernetes，如何在不牺牲开发敏捷性和运营独立性的前提下确保合规性？

例如，您可以执行以下策略：

* 所有镜像必须来自获得批准的存储库
* 所有入口主机名必须是全局唯一的
* 所有 Pod 必须有资源限制
* 所有命名空间都必须具有列出联系的标签

在接收请求被持久化为 Kubernetes 中的对象之前，Kubernetes 允许通过 [admission controller webhooks](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/) 将策略决策与 API 服务器分离，从而拦截这些请求。[Gatekeeper](https://github.com/open-policy-agent/gatekeeper) 创建的目的是使用户能够通过配置（而不是代码）自定义控制许可，并使用户了解集群的状态，而不仅仅是针对评估状态的单个对象，在这些对象准许加入的时候。Gatekeeper 是 Kubernetes 的一个可定制的许可 webhook ，它由 [Open Policy Agent (OPA)](https://www.openpolicyagent.org) 强制执行， OPA 是 Cloud Native 环境下的策略引擎，由 CNCF 主办。

## 发展 

在深入了解 Gatekeeper 的当前情况之前，让我们看一下 Gatekeeper 项目是如何发展的。

* Gatekeeper v1.0 - 使用 OPA 作为带有 kube-mgmt sidecar 的许可控制器，用来强制执行基于 configmap 的策略。这种方法实现了验证和转换许可控制。贡献方：Styra
* Gatekeeper v2.0 - 使用 Kubernetes 策略控制器作为许可控制器，OPA 和 kube-mgmt sidecar 实施基于 configmap 的策略。这种方法实现了验证和转换准入控制和审核功能。贡献方：Microsoft
 * Gatekeeper v3.0 - 准入控制器与 [OPA Constraint Framework](https://github.com/open-policy-agent/frameworks/tree/master/constraint) 集成在一起，用来实施基于 CRD 的策略，并可以可靠地共享已完成声明配置的策略。使用 kubebuilder 进行构建，实现了验证以及最终转换（待完成）为许可控制和审核功能。这样就可以为 [Rego](https://www.openpolicyagent.org/docs/latest/how-do-i-write-policies/) 策略创建策略模板，将策略创建为 CRD 并存储审核结果到策略 CRD 上。该项目是 Google，Microsoft，Red Hat 和 Styra 合作完成的。

![](/images/blog/2019-08-06-opa-gatekeeper/v3.png)
 
## Gatekeeper v3.0 的功能 

现在我们详细看一下 Gatekeeper 当前的状态，以及如何利用所有最新的功能。假设一个组织希望确保集群中的所有对象都有 department 信息，这些信息是对象标签的一部分。如何利用 Gatekeeper 完成这项需求？

### 验证许可控制 

在集群中所有 Gatekeeper 组件都 [安装](https://github.com/open-policy-agent/gatekeeper) 完成之后，只要集群中的资源进行创建、更新或删除，API 服务器将触发 Gatekeeper 准入 webhook 来处理准入请求。

在验证过程中，Gatekeeper 充当 API 服务器和 OPA 之间的桥梁。API 服务器将强制实施 OPA 执行的所有策略。

### 策略与 Constraint 

结合 OPA Constraint Framework，Constraint 是一个声明，表示作者希望系统满足给定的一系列要求。Constraint 都使用 Rego 编写，Rego 是声明性查询语言，OPA 用 Rego 来枚举违背系统预期状态的数据实例。所有 Constraint 都遵循逻辑 AND。假使有一个 Constraint 不满足，那么整个请求都将被拒绝。

在定义 Constraint 之前，您需要创建一个 Constraint Template，允许大家声明新的 Constraint。每个模板都描述了强制执行 Constraint 的 Rego 逻辑和 Constraint 的模式，其中包括 CRD 的模式和传递到 enforces 中的参数，就像函数的参数一样。              

例如，以下是一个 Constraint 模板 CRD，它的请求是在任意对象上显示某些标签。

```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
        listKind: K8sRequiredLabelsList
        plural: k8srequiredlabels
        singular: k8srequiredlabels
      validation:
        # Schema for the `parameters` field
        openAPIV3Schema:
          properties:
            labels:
              type: array
              items: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequiredlabels

        deny[{"msg": msg, "details": {"missing_labels": missing}}] {
          provided := {label | input.review.object.metadata.labels[label]}
          required := {label | label := input.parameters.labels[_]}
          missing := required - provided
          count(missing) > 0
          msg := sprintf("you must provide labels: %v", [missing])
        }
```

在集群中部署了 Constraint 模板后，管理员现在可以创建由 Constraint 模板定义的单个 Constraint CRD。例如，这里以下是一个 Constraint CRD，要求标签 `hr` 出现在所有命名空间上。

```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: ns-must-have-hr
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Namespace"]
  parameters:
    labels: ["hr"]
```

类似地，可以从同一个 Constraint 模板轻松地创建另一个 Constraint CRD，该 Constraint CRD 要求所有命名空间上都有 `finance` 标签。

```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: ns-must-have-finance
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Namespace"]
  parameters:
    labels: ["finance"]
```

如您所见，使用 Constraint framework，我们可以通过 Constraint 模板可靠地共享 rego，使用匹配字段定义执行范围，并为 Constraint 提供用户定义的参数，从而为每个 Constraint 创建自定义行为。

### 审核 

根据集群中强制执行的 Constraint，审核功能可定期评估复制的资源，并检测先前存在的错误配置。Gatekeeper 将审核结果存储为 `violations`，在相关 Constraint 的 `status` 字段中列出。

```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: ns-must-have-hr
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Namespace"]
  parameters:
    labels: ["hr"]
status:
  auditTimestamp: "2019-08-06T01:46:13Z"
  byPod:
  - enforced: true
    id: gatekeeper-controller-manager-0
  violations:
  - enforcementAction: deny
    kind: Namespace
    message: 'you must provide labels: {"hr"}'
    name: default
  - enforcementAction: deny
    kind: Namespace
    message: 'you must provide labels: {"hr"}'
    name: gatekeeper-system
  - enforcementAction: deny
    kind: Namespace
    message: 'you must provide labels: {"hr"}'
    name: kube-public
  - enforcementAction: deny
    kind: Namespace
    message: 'you must provide labels: {"hr"}'
    name: kube-system
```

### 数据复制 

审核要求将 Kubernetes 复制到 OPA 中，然后才能根据强制的 Constraint 对其进行评估。数据复制同样也需要 Constraint，这些 Constraint 需要访问集群中除评估对象之外的对象。例如，一个 Constraint 要强制确定入口主机名的唯一性，就必须有权访问集群中的所有其他入口。

对 Kubernetes 数据进行复制，请使用复制到 OPA 中的资源创建 sync config 资源。例如，下面的配置将所有命名空间和 Pod 资源复制到 OPA。

```yaml
apiVersion: config.gatekeeper.sh/v1alpha1
kind: Config
metadata:
  name: config
  namespace: "gatekeeper-system"
spec:
  sync:
    syncOnly:
      - group: ""
        version: "v1"
        kind: "Namespace"
      - group: ""
        version: "v1"
        kind: "Pod"
```

## 未来计划 

Gatekeeper 项目背后的社区将专注于提供转换许可控制，可以用来支持转换方案（例如：在创建新资源时使用 department 信息自动注释对象），支持外部数据以将集群外部环境加入到许可决策中，支持试运行以便在执行策略之前了解策略对集群中现有资源的影响，还有更多的审核功能。

如果您有兴趣了解更多有关该项目的信息，请查看 [Gatekeeper](https://github.com/open-policy-agent/gatekeeper) 存储库。如果您有兴趣帮助确定 Gatekeeper 的方向，请加入 [#kubernetes-policy](https://openpolicyagent.slack.com/messages/CDTN970AX) OPA Slack 频道，并加入我们的 [周会](https://docs.google.com/document/d/1A1-Q-1OMw3QODs1wT6eqfLTagcGmgzAJAjJihiO3T48/edit) 一同讨论开发、任务、用例等。 
