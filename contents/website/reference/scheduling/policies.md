---
title: 调度策略
content_type: concept
sitemap:
  priority: 0.2 # Scheduling priorities are deprecated
weight: 30
---



在 Kubernetes v1.23 版本之前，可以使用调度策略来指定 **predicates** 和 **priorities** 进程。
例如，可以通过运行 `kube-scheduler --policy-config-file <filename>` 或者
 `kube-scheduler --policy-configmap <ConfigMap>` 设置调度策略。

但是从 Kubernetes v1.23 版本开始，不再支持这种调度策略。
同样地也不支持相关的 `policy-config-file`、`policy-configmap`、`policy-configmap-namespace` 和 `use-legacy-policy-config` 标志。
你可以通过使用[调度配置](/zh-cn/docs/reference/scheduling/config/)来实现类似的行为。

## {{% heading "whatsnext" %}}


* 了解[调度](/zh-cn/docs/concepts/scheduling-eviction/kube-scheduler/)
* 了解 [kube-scheduler 配置](/zh-cn/docs/reference/scheduling/config/)
* 阅读 [kube-scheduler 配置参考 (v1)](/zh-cn/docs/reference/config-api/kube-scheduler-config.v1/)
