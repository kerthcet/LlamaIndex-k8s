---
title: kube-apiserver
content_type: tool-reference
weight: 30
auto_generated: true
---


## {{% heading "synopsis" %}}


Kubernetes API 服务器验证并配置 API 对象的数据，
这些对象包括 pods、services、replicationcontrollers 等。
API 服务器为 REST 操作提供服务，并为集群的共享状态提供前端，
所有其他组件都通过该前端进行交互。

```
kube-apiserver [flags]
```

## {{% heading "options" %}}

<table style="width: 100%; table-layout: fixed;">
<colgroup>
<col span="1" style="width: 10px;" />
<col span="1" />
</colgroup>
<tbody>

<tr>
<td colspan="2">--admission-control-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>包含准入控制配置的文件。</p>
</td>
</tr>

<tr>
<td colspan="2">--advertise-address string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>
向集群成员通知 apiserver 消息的 IP 地址。
这个地址必须能够被集群中其他成员访问。
如果 IP 地址为空，将会使用 --bind-address，
如果未指定 --bind-address，将会使用主机的默认接口地址。
</p>
</td>
</tr>

<tr>
</tr>
<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>聚合器拒绝将重定向响应转发回客户端。</p>
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
允许使用的指标标签到指标值的映射列表。键的格式为 &lt;MetricName&gt;,&lt;LabelName&gt;.
值的格式为 &lt;allowed_value&gt;,&lt;allowed_value&gt;...。
例如：<code>metric1,label1='v1,v2,v3', metric1,label2='v1,v2,v3' metric2,label1='v1,v2,v3'</code>。
</p></td>
</tr>

<tr>
<td colspan="2">--allow-privileged</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果为 true，将允许特权容器。[默认值=false]
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
启用针对 API 服务器的安全端口的匿名请求。
未被其他身份认证方法拒绝的请求被当做匿名请求。
匿名请求的用户名为 <code>system:anonymous</code>，
用户组名为 </code>system:unauthenticated</code>。
</td>
</tr>

<tr>
<td colspan="2">--api-audiences strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
API 的标识符。
服务帐户令牌验证者将验证针对 API 使用的令牌是否已绑定到这些受众中的至少一个。
如果配置了 <code>--service-account-issuer</code> 标志，但未配置此标志，
则此字段默认为包含发布者 URL 的单个元素列表。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
批处理和写入事件之前用于缓存事件的缓冲区大小。
仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr><tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
每个批次的最大大小。仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-batch-max-wait duration</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
强制写入尚未达到最大大小的批次之前要等待的时间。
仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-batch-throttle-burst int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果之前未使用 ThrottleQPS，则为同时发送的最大请求数。
仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-batch-throttle-enable</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
是否启用了批量限制。仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-batch-throttle-qps float</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
每秒的最大平均批次数。仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-compress</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
若设置了此标志，则被轮换的日志文件会使用 gzip 压缩。
</td>

</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
所保存的审计格式。
"legacy" 表示每行一个事件的文本格式。"json" 表示结构化的 JSON 格式。
已知格式为 legacy，json。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-maxage int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
根据文件名中编码的时间戳保留旧审计日志文件的最大天数。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-maxbackup int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
要保留的旧的审计日志文件个数上限。
将值设置为 0 表示对文件个数没有限制。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-maxsize int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
轮换之前，审计日志文件的最大大小（以兆字节为单位）。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用来发送审计事件的策略。
阻塞（blocking）表示发送事件应阻止服务器响应。
批处理（batch）会导致后端异步缓冲和写入事件。
已知的模式是批处理（batch），阻塞（blocking），严格阻塞（blocking-strict）。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-path string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果设置，则所有到达 API 服务器的请求都将记录到该文件中。
"-" 表示标准输出。
</td>
</tr>

<tr>
<td colspan="2">--audit-log-truncate-enabled</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
是否启用事件和批次截断。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
发送到下层后端的每批次的最大数据量。
实际的序列化大小可能会增加数百个字节。
如果一个批次超出此限制，则将其分成几个较小的批次。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
发送到下层后端的每批次的最大数据量。
如果事件的大小大于此数字，则将删除第一个请求和响应；
如果这样做没有减小足够大的程度，则将丢弃事件。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于对写入日志的审计事件执行序列化的 API 组和版本。
</td>
</tr>

<tr>
<td colspan="2">--audit-policy-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
定义审计策略配置的文件的路径。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
划分批次和写入之前用于存储事件的缓冲区大小。
仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
批次的最大大小。
仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
强制写入尚未达到最大大小的批处理之前要等待的时间。
仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果之前未使用 ThrottleQPS，同时发送的最大请求数。
仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
是否启用了批量限制。仅在批处理模式下使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
每秒的最大平均批次数。仅在批处理模式下使用。
</td>
</tr>

<tr>
<td colspan="2">--audit-webhook-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
定义审计 webhook 配置的 kubeconfig 格式文件的路径。
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
重试第一个失败的请求之前要等待的时间。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
发送审计事件的策略。
阻止（Blocking）表示发送事件应阻止服务器响应。
批处理（Batch）导致后端异步缓冲和写入事件。
已知的模式是批处理（batch），阻塞（blocking），严格阻塞（blocking-strict）。
</td>
</tr>

<tr>
<td colspan="2">--audit-webhook-truncate-enabled</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
是否启用事件和批处理截断。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
发送到下层后端的批次的最大数据量。
实际的序列化大小可能会增加数百个字节。
如果一个批次超出此限制，则将其分成几个较小的批次。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
发送到下层后端的批次的最大数据量。
如果事件的大小大于此数字，则将删除第一个请求和响应；
如果事件和事件的大小没有减小到一定幅度，则将丢弃事件。
</td>
</tr>

<tr>
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于序列化写入 Webhook 的审计事件的 API 组和版本。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
对来自 Webhook 令牌身份验证器的响应的缓存时间。
</td>
</tr>

<tr>
<td colspan="2">--authentication-token-webhook-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含 Webhook 配置的 kubeconfig 格式文件，用于进行令牌认证。
API 服务器将查询远程服务，以对持有者令牌进行身份验证。
</td>
</tr>

<tr>
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
与 Webhook 之间交换 authentication.k8s.io TokenReview 时使用的 API 版本。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
在安全端口上进行鉴权的插件的顺序列表。
逗号分隔的列表：AlwaysAllow、AlwaysDeny、ABAC、Webhook、RBAC、Node。
</td>
</tr>

<tr>
<td colspan="2">--authorization-policy-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含鉴权策略的文件，其内容为分行 JSON 格式，
在安全端口上与 --authorization-mode=ABAC 一起使用。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
对来自 Webhook 鉴权组件的 “授权（authorized）” 响应的缓存时间。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
对来自 Webhook 鉴权模块的 “未授权（unauthorized）” 响应的缓存时间。
</td>
</tr>

<tr>
<td colspan="2">--authorization-webhook-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含 Webhook 配置的文件，其格式为 kubeconfig，
与 --authorization-mode=Webhook 一起使用。
API 服务器将查询远程服务，以对 API 服务器的安全端口的访问执行鉴权。
</td>
</tr>

<tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
与 Webhook 之间交换 authorization.k8s.io SubjectAccessReview 时使用的 API 版本。
</td>
</tr>

<tr>
<td colspan="2">--azure-container-registry-config string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含 Azure 容器仓库配置信息的文件的路径。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用来监听 <code>--secure-port</code> 端口的 IP 地址。
集群的其余部分以及 CLI/web 客户端必须可以访问所关联的接口。
如果为空白或未指定地址（<tt>0.0.0.0</tt> 或 <tt>::</tt>），则将使用所有接口。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
TLS 证书所在的目录。
如果提供了 <code>--tls-cert-file</code> 和 <code>--tls-private-key-file</code>
标志值，则将忽略此标志。
</td>
</tr>

<tr>
<td colspan="2">--client-ca-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果已设置，则使用与客户端证书的 CommonName 对应的标识对任何出示由
client-ca 文件中的授权机构之一签名的客户端证书的请求进行身份验证。
</td>
</tr>

<tr>
<td colspan="2">--cloud-config string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
云厂商配置文件的路径。空字符串表示无配置文件。
</td>
</tr>

<tr>
<td colspan="2">--cloud-provider string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
云服务提供商。空字符串表示没有云厂商。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
在 GCE 防火墙中打开 CIDR，以进行第 7 层负载均衡流量代理和健康状况检查。
</td>
</tr>

<tr>
<td colspan="2">--contention-profiling</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果启用了性能分析，则启用阻塞分析。
</td>
</tr>

<tr>
<td colspan="2">--cors-allowed-origins strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>
CORS 允许的来源清单，以逗号分隔。
允许的来源可以是支持子域匹配的正则表达式。
如果此列表为空，则不会启用 CORS。
请确保每个表达式与整个主机名相匹配，方法是用'^'锚定开始或包括'//'前缀，同时用'$'锚定结束或包括':'端口分隔符后缀。
有效表达式的例子是'//example.com(:|$)'和'^https://example.com(:|$)'。
</p>
</td>
</tr>

<tr>
<td colspan="2">--debug-socket-path string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
使用位于给定路径的、未受保护的（无身份认证或鉴权的）UNIX 域套接字执行性能分析。
</p></td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
对污点 NotReady:NoExecute 的容忍时长（以秒计）。
默认情况下这一容忍度会被添加到尚未具有此容忍度的每个 pod 中。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
对污点 Unreachable:NoExecute 的容忍时长（以秒计）
默认情况下这一容忍度会被添加到尚未具有此容忍度的每个 pod 中。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
为 DeleteCollection 调用而产生的工作线程数。
这些用于加速名字空间清理。
</td>
</tr>

<tr>
<td colspan="2">--disable-admission-plugins strings</td>
</tr>

<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>
尽管位于默认启用的插件列表中，仍须被禁用的准入插件（NamespaceLifecycle、LimitRanger、ServiceAccount、TaintNodesByCondition、PodSecurity、Priority、DefaultTolerationSeconds、DefaultStorageClass、StorageObjectInUseProtection、PersistentVolumeClaimResize、RuntimeClass、CertificateApproval、CertificateSigning、ClusterTrustBundleAttest、CertificateSubjectRestriction、DefaultIngressClass、MutatingAdmissionWebhook、ValidatingAdmissionPolicy、ValidatingAdmissionWebhook、ResourceQuota）。
取值为逗号分隔的准入插件列表：AlwaysAdmit、AlwaysDeny、AlwaysPullImages、CertificateApproval、CertificateSigning、CertificateSubjectRestriction、ClusterTrustBundleAttest、DefaultIngressClass、DefaultStorageClass、DefaultTolerationSeconds、DenyServiceExternalIPs、EventRateLimit、ExtendedResourceToleration、ImagePolicyWebhook、LimitPodHardAntiAffinityTopology、LimitRanger、MutatingAdmissionWebhook、NamespaceAutoProvision、NamespaceExists、NamespaceLifecycle、NodeRestriction、OwnerReferencesPermissionEnforcement、PersistentVolumeClaimResize、PersistentVolumeLabel、PodNodeSelector、PodSecurity、PodTolerationRestriction、Priority、ResourceQuota、RuntimeClass、SecurityContextDeny、ServiceAccount、StorageObjectInUseProtection、TaintNodesByCondition、ValidatingAdmissionPolicy、ValidatingAdmissionWebhook。
该标志中插件的顺序无关紧要。
</p>
</td>
</tr>

<tr>
<td colspan="2">--disabled-metrics strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
此标志为行为不正确的度量指标提供一种处理方案。
你必须提供完全限定的指标名称才能将其禁止。
声明：禁用度量值的行为优先于显示已隐藏的度量值。
</td>
</tr>

<tr>
<td colspan="2">--egress-selector-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
带有 API 服务器出站选择器配置的文件。
</td>
</tr>

<tr>
<td colspan="2">--enable-admission-plugins strings</td>
</tr>

<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>
除了默认启用的插件（NamespaceLifecycle、LimitRanger、ServiceAccount、TaintNodesByCondition、PodSecurity、Priority、DefaultTolerationSeconds、DefaultStorageClass、StorageObjectInUseProtection、PersistentVolumeClaimResize、RuntimeClass、CertificateApproval、CertificateSigning、ClusterTrustBundleAttest、CertificateSubjectRestriction、DefaultIngressClass、MutatingAdmissionWebhook、ValidatingAdmissionPolicy、ValidatingAdmissionWebhook、ResourceQuota）之外要启用的准入插件。
取值为逗号分隔的准入插件列表：AlwaysAdmit、AlwaysDeny、AlwaysPullImages、CertificateApproval、CertificateSigning、CertificateSubjectRestriction、ClusterTrustBundleAttest、DefaultIngressClass、DefaultStorageClass、DefaultTolerationSeconds、DenyServiceExternalIPs、EventRateLimit、ExtendedResourceToleration、ImagePolicyWebhook、LimitPodHardAntiAffinityTopology、LimitRanger、MutatingAdmissionWebhook、NamespaceAutoProvision、NamespaceExists、NamespaceLifecycle、NodeRestriction、OwnerReferencesPermissionEnforcement、PersistentVolumeClaimResize、PersistentVolumeLabel、PodNodeSelector、PodSecurity、PodTolerationRestriction、Priority、ResourceQuota、RuntimeClass、SecurityContextDeny、ServiceAccount、StorageObjectInUseProtection、TaintNodesByCondition、ValidatingAdmissionPolicy、ValidatingAdmissionWebhook。该标志中插件的顺序无关紧要。
</p>
</td>
</tr>

<tr>
<td colspan="2">--enable-aggregator-routing</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
允许聚合器将请求路由到端点 IP 而非集群 IP。
</td>
</tr>

<tr>
<td colspan="2">--enable-bootstrap-token-auth</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
启用以允许将 "kube-system" 名字空间中类型为 "bootstrap.kubernetes.io/token"
的 Secret 用于 TLS 引导身份验证。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
启用通用垃圾收集器。必须与 kube-controller-manager 的相应标志同步。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果为 true 且启用了 <code>APIPriorityAndFairness</code> 特性门控，
则使用增强的处理程序替换 max-in-flight 处理程序，
以便根据优先级和公平性完成排队和调度。
</td>
</tr>

<tr>
<td colspan="2">--encryption-provider-config string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含加密提供程序配置信息的文件，用在 etcd 中所存储的 Secret 上。
</td>
</tr>

<tr>
<td colspan="2">--encryption-provider-config-automatic-reload</td>
</tr>
<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>
确定由 --encryption-provider-config 设置的文件是否应在磁盘内容更改时自动重新加载。
将此标志设置为 true 将禁用通过 API 服务器 healthz 端点来唯一地标识不同 KMS 插件的能力。
</p>
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
使用端点协调器（<code>master-count</code>、<code>lease</code> 或 <code>none</code>）。
<code>master-count</code> 已弃用，并将在未来版本中删除。
</td>
</tr>

<tr>
<td colspan="2">--etcd-cafile string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于保护 etcd 通信的 SSL 证书颁发机构文件。
</td>
</tr>

<tr>
<td colspan="2">--etcd-certfile string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于保护 etcd 通信的 SSL 证书文件。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
压缩请求的间隔。
如果为0，则禁用来自 API 服务器的压缩请求。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
针对每种类型的资源数量轮询 etcd 的频率。
0 值表示禁用度量值收集。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
轮询 etcd 和更新度量值的请求间隔。0 值表示禁用度量值收集。
</td>
</tr>

<tr>
检查 etcd 健康状况时使用的超时时长。
</td>
</tr>

<tr>
<td colspan="2">--etcd-keyfile string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于保护 etcd 通信的 SSL 密钥文件。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
要在 etcd 中所有资源路径之前添加的前缀。
</td>
</tr>

<tr>
<td colspan="2">
--etcd-readycheck-timeout 时长&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认值: 2s
</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
检查 etcd 是否就绪时使用的超时</p></td>
</tr>

<tr>
<td colspan="2">--etcd-servers strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
要连接的 etcd 服务器列表（<code>scheme://ip:port</code>），以逗号分隔。
</td>
</tr>

<tr>
<td colspan="2">--etcd-servers-overrides strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
etcd 服务器针对每个资源的重载设置，以逗号分隔。
单个替代格式：组/资源#服务器（group/resource#servers），
其中服务器是 URL，以分号分隔。
注意，此选项仅适用于编译进此服务器二进制文件的资源。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
事件的保留时长。
</td>
</tr>

<tr>
<td colspan="2">--external-hostname string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
为此主机生成外部化 UR L时要使用的主机名（例如 Swagger API 文档或 OpenID 发现）。
</td>
</tr>

<tr>
<td colspan="2">--feature-gates &lt;
逗号分隔的 'key=True|False' 键值对&gt;</td>
</tr>

<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;"><p>
一组 key=value 对，用来描述测试性/试验性功能的特性门控。可选项有：<br/>
APIListChunking=true|false (BETA - 默认值=true)<br/>
APIPriorityAndFairness=true|false (BETA - 默认值=true)<br/>
APIResponseCompression=true|false (BETA - 默认值=true)<br/>
APISelfSubjectReview=true|false (BETA - 默认值=true)<br/>
APIServerIdentity=true|false (BETA - 默认值=true)<br/>
APIServerTracing=true|false (BETA - 默认值=true)<br/>
AdmissionWebhookMatchConditions=true|false (ALPHA - 默认值=false)<br/>
AggregatedDiscoveryEndpoint=true|false (BETA - 默认值=true)<br/>
AllAlpha=true|false (ALPHA - 默认值=false)<br/>
AllBeta=true|false (BETA - 默认值=false)<br/>
AnyVolumeDataSource=true|false (BETA - 默认值=true)<br/>
AppArmor=true|false (BETA - 默认值=true)<br/>
CPUManagerPolicyAlphaOptions=true|false (ALPHA - 默认值=false)<br/>
CPUManagerPolicyBetaOptions=true|false (BETA - 默认值=true)<br/>
CPUManagerPolicyOptions=true|false (BETA - 默认值=true)<br/>
CSIMigrationPortworx=true|false (BETA - 默认值=false)<br/>
CSIMigrationRBD=true|false (ALPHA - 默认值=false)<br/>
CSINodeExpandSecret=true|false (BETA - 默认值=true)<br/>
CSIVolumeHealth=true|false (ALPHA - 默认值=false)<br/>
CloudControllerManagerWebhook=true|false (ALPHA - 默认值=false)<br/>
CloudDualStackNodeIPs=true|false (ALPHA - 默认值=false)<br/>
ClusterTrustBundle=true|false (ALPHA - 默认值=false)<br/>
ComponentSLIs=true|false (BETA - 默认值=true)<br/>
ContainerCheckpoint=true|false (ALPHA - 默认值=false)<br/>
ContextualLogging=true|false (ALPHA - 默认值=false)<br/>
CrossNamespaceVolumeDataSource=true|false (ALPHA - 默认值=false)<br/>
CustomCPUCFSQuotaPeriod=true|false (ALPHA - 默认值=false)<br/>
CustomResourceValidationExpressions=true|false (BETA - 默认值=true)<br/>
DisableCloudProviders=true|false (ALPHA - 默认值=false)<br/>
DisableKubeletCloudCredentialProviders=true|false (ALPHA - 默认值=false)<br/>
DynamicResourceAllocation=true|false (ALPHA - 默认值=false)<br/>
ElasticIndexedJob=true|false (BETA - 默认值=true)<br/>
EventedPLEG=true|false (BETA - 默认值=false)<br/>
ExpandedDNSConfig=true|false (BETA - 默认值=true)<br/>
ExperimentalHostUserNamespaceDefaulting=true|false (BETA - 默认值=false)<br/>
GracefulNodeShutdown=true|false (BETA - 默认值=true)<br/>
GracefulNodeShutdownBasedOnPodPriority=true|false (BETA - 默认值=true)<br/>
HPAContainerMetrics=true|false (BETA - 默认值=true)<br/>
HPAScaleToZero=true|false (ALPHA - 默认值=false)<br/>
HonorPVReclaimPolicy=true|false (ALPHA - 默认值=false)<br/>
IPTablesOwnershipCleanup=true|false (BETA - 默认值=true)<br/>
InPlacePodVerticalScaling=true|false (ALPHA - 默认值=false)<br/>
InTreePluginAWSUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginAzureDiskUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginAzureFileUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginGCEUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginOpenStackUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginPortworxUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginRBDUnregister=true|false (ALPHA - 默认值=false)<br/>
InTreePluginvSphereUnregister=true|false (ALPHA - 默认值=false)<br/>
JobPodFailurePolicy=true|false (BETA - 默认值=true)<br/>
JobReadyPods=true|false (BETA - 默认值=true)<br/>
KMSv2=true|false (BETA - 默认值=true)<br/>
KubeletInUserNamespace=true|false (ALPHA - 默认值=false)<br/>
KubeletPodResources=true|false (BETA - 默认值=true)<br/>
KubeletPodResourcesDynamicResources=true|false (ALPHA - 默认值=false)<br/>
KubeletPodResourcesGet=true|false (ALPHA - 默认值=false)<br/>
KubeletPodResourcesGetAllocatable=true|false (BETA - 默认值=true)<br/>
KubeletTracing=true|false (BETA - 默认值=true)<br/>
LegacyServiceAccountTokenTracking=true|false (BETA - 默认值=true)<br/>
LocalStorageCapacityIsolationFSQuotaMonitoring=true|false (ALPHA - 默认值=false)<br/>
LogarithmicScaleDown=true|false (BETA - 默认值=true)<br/>
LoggingAlphaOptions=true|false (ALPHA - 默认值=false)<br/>
LoggingBetaOptions=true|false (BETA - 默认值=true)<br/>
MatchLabelKeysInPodTopologySpread=true|false (BETA - 默认值=true)<br/>
MaxUnavailableStatefulSet=true|false (ALPHA - 默认值=false)<br/>
MemoryManager=true|false (BETA - 默认值=true)<br/>
MemoryQoS=true|false (ALPHA - 默认值=false)<br/>
MinDomainsInPodTopologySpread=true|false (BETA - 默认值=true)<br/>
MinimizeIPTablesRestore=true|false (BETA - 默认值=true)<br/>
MultiCIDRRangeAllocator=true|false (ALPHA - 默认值=false)<br/>
MultiCIDRServiceAllocator=true|false (ALPHA - 默认值=false)<br/>
NetworkPolicyStatus=true|false (ALPHA - 默认值=false)<br/>
NewVolumeManagerReconstruction=true|false (BETA - 默认值=true)<br/>
NodeInclusionPolicyInPodTopologySpread=true|false (BETA - 默认值=true)<br/>
NodeLogQuery=true|false (ALPHA - 默认值=false)<br/>
NodeOutOfServiceVolumeDetach=true|false (BETA - 默认值=true)<br/>
NodeSwap=true|false (ALPHA - 默认值=false)<br/>
OpenAPIEnums=true|false (BETA - 默认值=true)<br/>
PDBUnhealthyPodEvictionPolicy=true|false (BETA - 默认值=true)<br/>
PodAndContainerStatsFromCRI=true|false (ALPHA - 默认值=false)<br/>
PodDeletionCost=true|false (BETA - 默认值=true)<br/>
PodDisruptionConditions=true|false (BETA - 默认值=true)<br/>
PodHasNetworkCondition=true|false (ALPHA - 默认值=false)<br/>
PodSchedulingReadiness=true|false (BETA - 默认值=true)<br/>
ProbeTerminationGracePeriod=true|false (BETA - 默认值=true)<br/>
ProcMountType=true|false (ALPHA - 默认值=false)<br/>
ProxyTerminatingEndpoints=true|false (BETA - 默认值=true)<br/>
QOSReserved=true|false (ALPHA - 默认值=false)<br/>
ReadWriteOncePod=true|false (BETA - 默认值=true)<br/>
RecoverVolumeExpansionFailure=true|false (ALPHA - 默认值=false)<br/>
RemainingItemCount=true|false (BETA - 默认值=true)<br/>
RetroactiveDefaultStorageClass=true|false (BETA - 默认值=true)<br/>
RotateKubeletServerCertificate=true|false (BETA - 默认值=true)<br/>
SELinuxMountReadWriteOncePod=true|false (BETA - 默认值=true)<br/>
SecurityContextDeny=true|false (ALPHA - 默认值=false)<br/>
ServiceNodePortStaticSubrange=true|false (ALPHA - 默认值=false)<br/>
SizeMemoryBackedVolumes=true|false (BETA - 默认值=true)<br/>
StableLoadBalancerNodeSet=true|false (BETA - 默认值=true)<br/>
StatefulSetAutoDeletePVC=true|false (BETA - 默认值=true)<br/>
StatefulSetStartOrdinal=true|false (BETA - 默认值=true)<br/>
StorageVersionAPI=true|false (ALPHA - 默认值=false)<br/>
StorageVersionHash=true|false (BETA - 默认值=true)<br/>
TopologyAwareHints=true|false (BETA - 默认值=true)<br/>
TopologyManagerPolicyAlphaOptions=true|false (ALPHA - 默认值=false)<br/>
TopologyManagerPolicyBetaOptions=true|false (BETA - 默认值=false)<br/>
TopologyManagerPolicyOptions=true|false (ALPHA - 默认值=false)<br/>
UserNamespacesStatelessPodsSupport=true|false (ALPHA - 默认值=false)<br/>
ValidatingAdmissionPolicy=true|false (ALPHA - 默认值=false)<br/>
VolumeCapacityPriority=true|false (ALPHA - 默认值=false)<br/>
WatchList=true|false (ALPHA - 默认值=false)<br/>
WinDSR=true|false (ALPHA - 默认值=false)<br/>
WinOverlay=true|false (BETA - 默认值=true)<br/>
WindowsHostNetwork=true|false (ALPHA - 默认值=true)
</p>
</td>
</tr>

<tr>
<td colspan="2">--goaway-chance float</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
为防止 HTTP/2 客户端卡在单个 API 服务器上，随机关闭某连接（GOAWAY）。
客户端的其他运行中请求不会受到影响。被关闭的客户端将重新连接，
重新被负载均衡后可能会与其他 API 服务器开始通信。
此参数设置将被发送 GOAWAY 指令的请求的比例。
只有一个 API 服务器或不使用负载均衡器的集群不应启用此特性。
最小值为 0（关闭），最大值为 .02（1/50 请求）；建议使用 .001（1/1000）。
</td>
</tr>

<tr>
<td colspan="2">-h, --help</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
kube-apiserver 的帮助命令
</td>
</tr>

<tr>
<td colspan="2">--http2-max-streams-per-connection int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
服务器为客户端提供的 HTTP/2 连接中最大流数的限制。
零表示使用 GoLang 的默认值。
</td>
</tr>

<tr>
<td colspan="2">--kubelet-certificate-authority string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
证书颁发机构的证书文件的路径。
</td>
</tr>

<tr>
<td colspan="2">--kubelet-client-certificate string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
TLS 的客户端证书文件的路径。
</td>
</tr>

<tr>
<td colspan="2">--kubelet-client-key string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
TLS 客户端密钥文件的路径。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于 kubelet 连接的首选 NodeAddressTypes 列表。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
kubelet 操作超时时间。
</td>
</tr>

<tr>
<td colspan="2">--kubernetes-service-node-port int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果非零，那么 Kubernetes 主服务（由 apiserver 创建/维护）将是 NodePort 类型，
使用此字段值作为端口值。
如果为零，则 Kubernetes 主服务将为 ClusterIP 类型。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
每个租约被重用的时长。
如果此值比较低，可以避免大量对象重用此租约。
注意，如果此值过小，可能导致存储层出现性能问题。
</td>
</tr>

<tr>
<td colspan="2">--livez-grace-period duration</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
此选项代表 API 服务器完成启动序列并生效所需的最长时间。
从 API 服务器的启动时间到这段时间为止，
<tt>/livez</tt> 将假定未完成的启动后钩子将成功完成，因此返回 true。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
两次日志刷新之间的最大秒数。
</td>
</tr>

<tr>
</tr>
<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>
设置日志格式。允许的格式：&quot;text&quot;。
</p>
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
已废弃：应该从其中将 Kubernetes 主服务注入到 Pod 中的名字空间。
</td>
</tr>

<tr>
<td colspan="2">--max-connection-bytes-per-sec int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果不为零，则将每个用户连接的带宽限制为此数值（字节数/秒）。
当前仅适用于长时间运行的请求。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果 --enable-priority-and-fairness 为 true，那么此值和 --max-requests-inflight
的和将确定服务器的总并发限制（必须是正数）。
否则，该值限制同时运行的变更类型的请求的个数上限。0 表示无限制。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果 --enable-priority-and-fairness 为 true，那么此值和 --max-mutating-requests-inflight
的和将确定服务器的总并发限制（必须是正数）。
否则，该值限制进行中非变更类型请求的最大个数，零表示无限制。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
可选字段，表示处理程序在请求超时前，必须保持连接处于打开状态的最小秒数。
当前只对监听（Watch）请求的处理程序有效。
Watch 请求的处理程序会基于这个值选择一个随机数作为连接超时值，
以达到分散负载的目的。
</td>
</tr>

<tr>
<td colspan="2">--oidc-ca-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果设置该值，将会使用 oidc-ca-file 中的机构之一对 OpenID 服务的证书进行验证，
否则将会使用主机的根 CA 对其进行验证。
</td>
</tr>

<tr>
<td colspan="2">--oidc-client-id string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
OpenID 连接客户端的要使用的客户 ID，如果设置了 oidc-issuer-url，则必须设置这个值。
</td>
</tr>

<tr>
<td colspan="2">--oidc-groups-claim string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果提供该值，这个自定义 OpenID 连接声明将被用来设定用户组。
该声明值需要是一个字符串或字符串数组。
此标志为实验性的，请查阅身份认证相关文档进一步了解详细信息。
</td>
</tr>

<tr>
<td colspan="2">--oidc-groups-prefix string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果提供了此值，则所有组都将以该值作为前缀，以防止与其他身份认证策略冲突。
</td>
</tr>

<tr>
<td colspan="2">--oidc-issuer-url string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
OpenID 颁发者 URL，只接受 HTTPS 方案。
如果设置该值，它将被用于验证 OIDC JSON Web Token(JWT)。
</td>
</tr>

<tr>
<td colspan="2">--oidc-required-claim &lt;逗号分隔的 'key=value' 键值对列表&gt;</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
描述 ID 令牌中必需声明的键值对。
如果设置此值，则会验证 ID 令牌中存在与该声明匹配的值。
重复此标志以指定多个声明。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
允许的 JOSE 非对称签名算法的逗号分隔列表。
具有收支持 "alg" 标头值的 JWTs 有：RS256、RS384、RS512、ES256、ES384、ES512、PS256、PS384、PS512。
取值依据 RFC 7518 https://tools.ietf.org/html/rfc7518#section-3.1 定义。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
要用作用户名的 OpenID 声明。
请注意，除默认声明（"sub"）以外的其他声明不能保证是唯一且不可变的。
此标志是实验性的，请参阅身份认证文档以获取更多详细信息。
</td>
</tr>

<tr>
<td colspan="2">--oidc-username-prefix string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果提供，则所有用户名都将以该值作为前缀。
如果未提供，则除 "email" 之外的用户名声明都会添加颁发者 URL 作为前缀，以避免冲突。
要略过添加前缀处理，请设置值为 "-"。
</td>
</tr>

<tr>
<td colspan="2">--permit-address-sharing</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
若此标志为 true，则使用 <tt>SO_REUSEADDR</tt> 来绑定端口。
这样设置可以同时绑定到用通配符表示的类似 0.0.0.0 这种 IP 地址，
以及特定的 IP 地址。也可以避免等待内核释放 <tt>TIME_WAIT</tt> 状态的套接字。[默认值=false]
</p></td>
</tr>

<tr>
<td colspan="2">--permit-port-sharing</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果为 true，则在绑定端口时将使用 <tt>SO_REUSEPORT</tt>，
这样多个实例可以绑定到同一地址和端口上。[默认值=false]
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
通过 Web 接口 <code>host:port/debug/pprof/</code> 启用性能分析。
</td>
</tr>

<tr>
<td colspan="2">--proxy-client-cert-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
当必须调用外部程序以处理请求时，用于证明聚合器或者 kube-apiserver 的身份的客户端证书。
包括代理转发到用户 api-server 的请求和调用 Webhook 准入控制插件的请求。
Kubernetes 期望此证书包含来自于 --requestheader-client-ca-file 标志中所给 CA 的签名。
该 CA 在 kube-system 命名空间的 "extension-apiserver-authentication" ConfigMap 中公开。
从 kube-aggregator 收到调用的组件应该使用该 CA 进行各自的双向 TLS 验证。
</td>
</tr>

<tr>
<td colspan="2">--proxy-client-key-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
当必须调用外部程序来处理请求时，用来证明聚合器或者 kube-apiserver 的身份的客户端私钥。
这包括代理转发给用户 api-server 的请求和调用 Webhook 准入控制插件的请求。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
可选字段，指示处理程序在超时之前必须保持打开请求的持续时间。
这是请求的默认请求超时，但对于特定类型的请求，可能会被
<code>--min-request-timeout</code>等标志覆盖。
</td>
</tr>

<tr>
<td colspan="2">--requestheader-allowed-names strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
此值为客户端证书通用名称（Common Name）的列表；表中所列的表项可以用来提供用户名，
方式是使用 <code>--requestheader-username-headers</code> 所指定的头部。
如果为空，能够通过 <code>--requestheader-client-ca-file</code> 中机构
认证的客户端证书都是被允许的。
</td>
</tr>

<tr>
<td colspan="2">--requestheader-client-ca-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
在信任请求头中以 <code>--requestheader-username-headers</code> 指示的用户名之前，
用于验证接入请求中客户端证书的根证书包。
警告：一般不要假定传入请求已被授权。
</td>
</tr>

<tr>
<td colspan="2">--requestheader-extra-headers-prefix strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于查验请求头部的前缀列表。建议使用 <code>X-Remote-Extra-</code>。
</td>
</tr>

<tr>
<td colspan="2">--requestheader-group-headers strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于查验用户组的请求头部列表。建议使用 <code>X-Remote-Group</code>。
</td>
</tr>

<tr>
<td colspan="2">--requestheader-username-headers strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
用于查验用户名的请求头部字段列表。建议使用 <code>X-Remote-User</code>。
</td>
</tr>

<tr>
<td colspan="2">--runtime-config &lt;逗号分隔的 'key=value' 对列表&gt;</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
一组启用或禁用内置 API 的键值对。支持的选项包括：
<br/>v1=true|false（针对核心 API 组）
<br/>&lt;group&gt;/&lt;version&gt;=true|false（针对特定 API 组和版本，例如：apps/v1=true） 
<br/>api/all=true|false 控制所有 API 版本
<br/>api/ga=true|false 控制所有 v[0-9]+ API 版本
<br/>api/beta=true|false 控制所有 v[0-9]+beta[0-9]+ API 版本
<br/>api/alpha=true|false 控制所有 v[0-9]+alpha[0-9]+ API 版本
<br/>api/legacy 已弃用，并将在以后的版本中删除
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
带身份验证和鉴权机制的 HTTPS 服务端口。
不能用 0 关闭。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
在生成令牌时，启用投射服务帐户到期时间扩展，
这有助于从旧版令牌安全地过渡到绑定的服务帐户令牌功能。
如果启用此标志，则准入插件注入的令牌的过期时间将延长至 1 年，以防止过渡期间发生意外故障，
并忽略 service-account-max-token-expiration 的值。
</td>
</tr>

<tr>
<td colspan="2">--service-account-issuer strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
服务帐号令牌颁发者的标识符。
颁发者将在已颁发令牌的 "iss" 声明中检查此标识符。
此值为字符串或 URI。
如果根据 OpenID Discovery 1.0 规范检查此选项不是有效的 URI，则即使特性门控设置为 true，
ServiceAccountIssuerDiscovery 功能也将保持禁用状态。
强烈建议该值符合 OpenID 规范： https://openid.net/specs/openid-connect-discovery-1_0.html 。
实践中，这意味着 service-account-issuer 取值必须是 HTTPS URL。
还强烈建议此 URL 能够在 {service-account-issuer}/.well-known/openid-configuration
处提供 OpenID 发现文档。
当此值被多次指定时，第一次的值用于生成令牌，所有的值用于确定接受哪些发行人。
</td>
</tr>

<tr>
<td colspan="2">--service-account-jwks-uri string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
覆盖 <code>/.well-known/openid-configuration</code> 提供的发现文档中 JSON Web 密钥集的 URI。
如果发现文档和密钥集是通过 API 服务器外部
（而非自动检测到或被外部主机名覆盖）之外的 URL 提供给依赖方的，则此标志很有用。
</td>
</tr>

<tr>
<td colspan="2">--service-account-key-file strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含 PEM 编码的 x509 RSA 或 ECDSA 私钥或公钥的文件，用于验证 ServiceAccount 令牌。
指定的文件可以包含多个键，并且可以使用不同的文件多次指定标志。
如果未指定，则使用 <code>--tls-private-key-file</code>。
提供 <code>--service-account-signing-key-file</code> 时必须指定。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果为 true，则在身份认证时验证 etcd 中是否存在 ServiceAccount 令牌。
</td>
</tr>

<tr>
<td colspan="2">--service-account-max-token-expiration duration</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
服务帐户令牌发布者创建的令牌的最长有效期。
如果请求有效期大于此值的有效令牌请求，将使用此值的有效期颁发令牌。
</td>
</tr>

<tr>
<td colspan="2">--service-account-signing-key-file string</td>
</tr>
<tr>

<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含服务帐户令牌颁发者当前私钥的文件的路径。
颁发者将使用此私钥签署所颁发的 ID 令牌。
</td>
</tr>

<tr>
<td colspan="2">--service-cluster-ip-range string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
CIDR 表示的 IP 范围用来为服务分配集群 IP。
此地址不得与指定给节点或 Pod 的任何 IP 范围重叠。
最多允许两个双栈 CIDR。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>保留给具有 NodePort 可见性的服务的端口范围。
不得与节点上的临时端口范围重叠。
例如："30000-32767"。范围的两端都包括在内。</p>
</td>
</tr>

<tr>
<td colspan="2">--show-hidden-metrics-for-version string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
你要显示隐藏指标的先前版本。仅先前的次要版本有意义，不允许其他值。
格式为 &lt;major&gt;.&lt;minor&gt;，例如："1.16"。
这种格式的目的是确保你有机会注意到下一个版本是否隐藏了其他指标，
而不是在此之后将它们从发行版中永久删除时感到惊讶。
</td>
</tr>

<tr>
<td colspan="2">--shutdown-delay-duration duration</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
延迟终止时间。在此期间，服务器将继续正常处理请求。
端点 /healthz 和 /livez 将返回成功，但是 /readyz 立即返回失败。
在此延迟过去之后，将开始正常终止。
这可用于允许负载均衡器停止向该服务器发送流量。
</td>
</tr>

<tr>
<td colspan="2">--shutdown-send-retry-after</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
值为 true 表示 HTTP 服务器将继续监听直到耗尽所有非长时间运行的请求，
在此期间，所有传入请求将被拒绝，状态码为 429，响应头为 &quot;Retry-After&quot;，
此外，设置 &quot;Connection: close&quot; 响应头是为了在空闲时断开 TCP 链接。
</td>
</tr>

<tr>
<td colspan="2">--shutdown-watch-termination-grace-period duration</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
此选项如果被设置了，则表示 API 服务器体面关闭服务器窗口内，等待活跃的监听请求耗尽的最长宽限期。
</p></td>
</tr>

<tr>
<td colspan="2">--storage-backend string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
持久化存储后端。选项："etcd3"（默认）。
</td>
</tr>

<tr>
</tr>
<tr>
<td>
</td>
<td style="line-height: 130%; word-wrap: break-word;">
<p>
用于在存储中存储对象的媒体类型。某些资源或存储后端可能仅支持特定的媒体类型，并且将忽略此设置。
支持的媒体类型：[application/json, application/yaml, application/vnd.kubernetes.protobuf]
</p>
</td>
</tr>

<tr>
<td colspan="2">--strict-transport-security-directives strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;"><p>
为 HSTS 所设置的指令列表，用逗号分隔。
如果此列表为空，则不会添加 HSTS 指令。
例如：'max-age=31536000,includeSubDomains,preload'
</p></td>
</tr>

<tr>
<td colspan="2">--tls-cert-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含用于 HTTPS 的默认 x509 证书的文件。（CA 证书（如果有）在服务器证书之后并置）。
如果启用了 HTTPS 服务，并且未提供 <code>--tls-cert-file</code> 和
<code>--tls-private-key-file</code>，
为公共地址生成一个自签名证书和密钥，并将其保存到 <code>--cert-dir</code> 指定的目录中。
</td>
</tr>

<tr>
<td colspan="2">--tls-cipher-suites strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
服务器的密码套件的列表，以逗号分隔。如果省略，将使用默认的 Go 密码套件。
<br/>首选值：
TLS_AES_128_GCM_SHA256、TLS_AES_256_GCM_SHA384、TLS_CHACHA20_POLY1305_SHA256、TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA、
TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256、TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA、TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384、TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305、TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256、TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA、TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256、TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA、TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384、TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305、TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256、TLS_RSA_WITH_AES_128_CBC_SHA、TLS_RSA_WITH_AES_128_GCM_SHA256、TLS_RSA_WITH_AES_256_CBC_SHA、TLS_RSA_WITH_AES_256_GCM_SHA384。
不安全的值有：
TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256、TLS_ECDHE_ECDSA_WITH_RC4_128_SHA、TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA、TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256、TLS_ECDHE_RSA_WITH_RC4_128_SHA、TLS_RSA_WITH_3DES_EDE_CBC_SHA、TLS_RSA_WITH_AES_128_CBC_SHA256、TLS_RSA_WITH_RC4_128_SHA。
</td>
</tr>

<tr>
<td colspan="2">--tls-min-version string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
支持的最低 TLS 版本。可能的值：VersionTLS10，VersionTLS11，VersionTLS12，VersionTLS13
</td>
</tr>

<tr>
<td colspan="2">--tls-private-key-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含匹配 <code>--tls-cert-file</code> 的 x509 证书私钥的文件。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
一对 x509 证书和私钥文件路径，（可选）后缀为全限定域名的域名模式列表，可以使用带有通配符的前缀。
域模式也允许使用 IP 地址，但仅当 apiserver 对客户端请求的IP地址具有可见性时，才应使用 IP。
如果未提供域模式，则提取证书的名称。
非通配符匹配优先于通配符匹配，显式域模式优先于提取出的名称。
对于多个密钥/证书对，请多次使用 <code>--tls-sni-cert-key</code>。
示例："example.crt,example.key" 或 "foo.crt,foo.key:\*.foo.com,foo.com"。
</td>
</tr>

<tr>
<td colspan="2">--token-auth-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
如果设置该值，这个文件将被用于通过令牌认证来保护 API 服务的安全端口。
</td>
</tr>

<tr>
<td colspan="2">--tracing-config-file string</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
包含 API 服务器跟踪配置的文件。
</td>
</tr>

<tr>
<td colspan="2">-v, --v int</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
日志级别详细程度的数字。
</td>
</tr>

<tr>
<td colspan="2">--version version[=true]</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
打印版本信息并退出
</td>
</tr>

<tr>
<td colspan="2">--vmodule pattern=N,...</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
以逗号分隔的 <code>pattern=N</code> 设置列表，用于文件过滤的日志记录（仅适用于 text 日志格式）。
</td>
</tr>

<tr>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
在 API 服务器中启用监视缓存。
</td>
</tr>

<tr>
<td colspan="2">--watch-cache-sizes strings</td>
</tr>
<tr>
<td></td><td style="line-height: 130%; word-wrap: break-word;">
<p>某些资源（Pod、Node 等）的监视缓存大小设置，以逗号分隔。
每个资源对应的设置格式：<code>resource[.group]#size</code>，其中
<code>resource</code> 为小写复数（无版本），
对于 apiVersion v1（旧版核心 API）的资源要省略 <code>group</code>，
对其它资源要给出 <code>group</code>；<code>size 为一个数字</code>。
此选项仅对 API 服务器中的内置资源生效，对 CRD 定义的资源或从外部服务器接入的资源无效。
启用 <code>watch-cache</code> 时仅查询此选项。
这里能生效的 size 设置只有 0，意味着禁用关联资源的 <code>watch-cache</code>。
所有的非零值都等效，意味着不禁用该资源的<code>watch-cache</code>。</p>


</td>
</tr>

</tbody>
</table>