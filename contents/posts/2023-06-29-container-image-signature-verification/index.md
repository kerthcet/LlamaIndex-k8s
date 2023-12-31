---
layout: blog
title: "在 CRI 运行时内验证容器镜像签名"
date: 2023-06-29
slug: container-image-signature-verification
---

**作者:** Sascha Grunert

**译者:** [Michael Yao](https://github.com/windsonsea) (DaoCloud)

Kubernetes 社区自 v1.24 版本开始对基于容器镜像的工件进行签名。在 v1.26 中，
[相应的增强特性][kep]从 `alpha` 进阶至 `beta`，引入了针对二进制工件的签名。
其他项目也采用了类似的方法，为其发布版本提供镜像签名。这意味着这些项目要么使用 GitHub actions
在自己的 CI/CD 流程中创建签名，要么依赖于 Kubernetes 的[镜像推广][promo]流程，
通过向 [k/k8s.io][k8s.io] 仓库提交 PR 来自动签名镜像。
使用此流程的前提要求是项目必须属于 `kubernetes` 或 `kubernetes-sigs` GitHub 组织，
这样能够利用社区基础设施将镜像推送到暂存桶中。

[kep]: https://github.com/kubernetes/enhancements/issues/3031
[promo]: https://github.com/kubernetes-sigs/promo-tools/blob/e2b96dd/docs/image-promotion.md
[k8s.io]: https://github.com/kubernetes/k8s.io/tree/4b95cc2/k8s.gcr.io

假设一个项目现在生成了已签名的容器镜像工件，那么如何实际验证这些签名呢？
你可以按照 [Kubernetes 官方文档][docs]所述来手动验证。但是这种方式的问题在于完全没有自动化，
应仅用于测试目的。在生产环境中，[sigstore policy-controller][policy-controller]
这样的工具有助于进行自动化处理。这些工具使用[自定义资源定义（CRD）][crd]提供了更高级别的 API，
并且利用集成的[准入控制器和 Webhook][admission]来验证签名。

[docs]: /zh-cn/docs/tasks/administer-cluster/verify-signed-artifacts/#verifying-image-signatures
[policy-controller]: https://docs.sigstore.dev/policy-controller/overview
[crd]: /zh-cn/docs/concepts/extend-kubernetes/api-extension/custom-resources
[admission]: /zh-cn/docs/reference/access-authn-authz/admission-controllers

基于准入控制器的验证的一般使用流程如下：

{{< figure src="/blog/2023/06/29/container-image-signature-verification/flow.svg" alt="创建一个策略的实例，并对命名空间添加注解以验证签名。然后创建 Pod。控制器会评估策略，如果评估通过，则根据需要执行镜像拉取。如果策略评估失败，则不允许该 Pod 运行。" >}}

这种架构的一个主要优点是简单：集群中的单个实例会先验证签名，然后才在节点上的容器运行时中执行镜像拉取操作，
镜像拉取是由 kubelet 触发的。这个优点也带来了分离的问题：应拉取容器镜像的节点不一定是执行准入控制的节点。
这意味着如果控制器受到攻击，那么无法在整个集群范围内强制执行策略。

解决此问题的一种方式是直接在与[容器运行时接口（CRI）][cri]兼容的容器运行时中进行策略评估。
这种运行时直接连接到节点上的 [kubelet][kubelet]，执行拉取镜像等所有任务。
[CRI-O][cri-o] 是可用的运行时之一，将在 v1.28 中完全支持容器镜像签名验证。

[cri]: /docs/concepts/architecture/cri
[kubelet]: /docs/reference/command-line-tools-reference/kubelet
[cri-o]: https://github.com/cri-o/cri-o

容器运行时是如何工作的呢？CRI-O 会读取一个名为 [`policy.json`][policy.json] 的文件，
其中包含了为容器镜像定义的所有规则。例如，你可以定义一个策略，
只允许带有以下标记或摘要的已签名镜像 `quay.io/crio/signed`：

[policy.json]: https://github.com/containers/image/blob/b3e0ba2/docs/containers-policy.json.5.md#sigstoresigned

```json
{
  "default": [{ "type": "reject" }],
  "transports": {
    "docker": {
      "quay.io/crio/signed": [
        {
          "type": "sigstoreSigned",
          "signedIdentity": { "type": "matchRepository" },
          "fulcio": {
            "oidcIssuer": "https://github.com/login/oauth",
            "subjectEmail": "sgrunert@redhat.com",
            "caData": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUI5ekNDQVh5Z0F3SUJBZ0lVQUxaTkFQRmR4SFB3amVEbG9Ed3lZQ2hBTy80d0NnWUlLb1pJemowRUF3TXcKS2pFVk1CTUdBMVVFQ2hNTWMybG5jM1J2Y21VdVpHVjJNUkV3RHdZRFZRUURFd2h6YVdkemRHOXlaVEFlRncweQpNVEV3TURjeE16VTJOVGxhRncwek1URXdNRFV4TXpVMk5UaGFNQ294RlRBVEJnTlZCQW9UREhOcFozTjBiM0psCkxtUmxkakVSTUE4R0ExVUVBeE1JYzJsbmMzUnZjbVV3ZGpBUUJnY3Foa2pPUFFJQkJnVXJnUVFBSWdOaUFBVDcKWGVGVDRyYjNQUUd3UzRJYWp0TGszL09sbnBnYW5nYUJjbFlwc1lCcjVpKzR5bkIwN2NlYjNMUDBPSU9aZHhleApYNjljNWlWdXlKUlErSHowNXlpK1VGM3VCV0FsSHBpUzVzaDArSDJHSEU3U1hyazFFQzVtMVRyMTlMOWdnOTJqCll6QmhNQTRHQTFVZER3RUIvd1FFQXdJQkJqQVBCZ05WSFJNQkFmOEVCVEFEQVFIL01CMEdBMVVkRGdRV0JCUlkKd0I1ZmtVV2xacWw2ekpDaGt5TFFLc1hGK2pBZkJnTlZIU01FR0RBV2dCUll3QjVma1VXbFpxbDZ6SkNoa3lMUQpLc1hGK2pBS0JnZ3Foa2pPUFFRREF3TnBBREJtQWpFQWoxbkhlWFpwKzEzTldCTmErRURzRFA4RzFXV2cxdENNCldQL1dIUHFwYVZvMGpoc3dlTkZaZ1NzMGVFN3dZSTRxQWpFQTJXQjlvdDk4c0lrb0YzdlpZZGQzL1Z0V0I1YjkKVE5NZWE3SXgvc3RKNVRmY0xMZUFCTEU0Qk5KT3NRNHZuQkhKCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0="
          },
          "rekorPublicKeyData": "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUZrd0V3WUhLb1pJemowQ0FRWUlLb1pJemowREFRY0RRZ0FFMkcyWSsydGFiZFRWNUJjR2lCSXgwYTlmQUZ3cgprQmJtTFNHdGtzNEwzcVg2eVlZMHp1ZkJuaEM4VXIvaXk1NUdoV1AvOUEvYlkyTGhDMzBNOStSWXR3PT0KLS0tLS1FTkQgUFVCTElDIEtFWS0tLS0tCg=="
        }
      ]
    }
  }
}
```

CRI-O 必须被启动才能将策略用作全局的可信源：

```console
> sudo crio --log-level debug --signature-policy ./policy.json
```

CRI-O 现在可以在验证镜像签名的同时拉取镜像。例如，可以使用 [`crictl`（cri-tools）][cri-tools]
来完成此操作：

[cri-tools]: https://github.com/kubernetes-sigs/cri-tools

```console
> sudo crictl -D pull quay.io/crio/signed
DEBU[…] get image connection
DEBU[…] PullImageRequest: &PullImageRequest{Image:&ImageSpec{Image:quay.io/crio/signed,Annotations:map[string]string{},},Auth:nil,SandboxConfig:nil,}
DEBU[…] PullImageResponse: &PullImageResponse{ImageRef:quay.io/crio/signed@sha256:18b42e8ea347780f35d979a829affa178593a8e31d90644466396e1187a07f3a,}
Image is up to date for quay.io/crio/signed@sha256:18b42e8ea347780f35d979a829affa178593a8e31d90644466396e1187a07f3a
```

CRI-O 的调试日志也会表明签名已成功验证：

```console
DEBU[…] IsRunningImageAllowed for image docker:quay.io/crio/signed:latest
DEBU[…]  Using transport "docker" specific policy section quay.io/crio/signed
DEBU[…] Reading /var/lib/containers/sigstore/crio/signed@sha256=18b42e8ea347780f35d979a829affa178593a8e31d90644466396e1187a07f3a/signature-1
DEBU[…] Looking for sigstore attachments in quay.io/crio/signed:sha256-18b42e8ea347780f35d979a829affa178593a8e31d90644466396e1187a07f3a.sig
DEBU[…] GET https://quay.io/v2/crio/signed/manifests/sha256-18b42e8ea347780f35d979a829affa178593a8e31d90644466396e1187a07f3a.sig
DEBU[…] Content-Type from manifest GET is "application/vnd.oci.image.manifest.v1+json"
DEBU[…] Found a sigstore attachment manifest with 1 layers
DEBU[…] Fetching sigstore attachment 1/1: sha256:8276724a208087e73ae5d9d6e8f872f67808c08b0acdfdc73019278807197c45
DEBU[…] Downloading /v2/crio/signed/blobs/sha256:8276724a208087e73ae5d9d6e8f872f67808c08b0acdfdc73019278807197c45
DEBU[…] GET https://quay.io/v2/crio/signed/blobs/sha256:8276724a208087e73ae5d9d6e8f872f67808c08b0acdfdc73019278807197c45
DEBU[…]  Requirement 0: allowed
DEBU[…] Overall: allowed
```

策略中定义的 `oidcIssuer` 和 `subjectEmail` 等所有字段都必须匹配，
而 `fulcio.caData` 和 `rekorPublicKeyData` 是来自上游 [fulcio（OIDC PKI）][fulcio]
和 [rekor（透明日志）][rekor] 实例的公钥。

[fulcio]: https://github.com/sigstore/fulcio
[rekor]: https://github.com/sigstore/rekor

这意味着如果你现在将策略中的 `subjectEmail` 作废，例如更改为 `wrong@mail.com`：

```console
> jq '.transports.docker."quay.io/crio/signed"[0].fulcio.subjectEmail = "wrong@mail.com"' policy.json > new-policy.json
> mv new-policy.json policy.json
```

然后移除镜像，因为此镜像已存在于本地：

```console
> sudo crictl rmi quay.io/crio/signed
```

现在当你拉取镜像时，CRI-O 将报错所需的 email 是错的：

```console
> sudo crictl pull quay.io/crio/signed
FATA[…] pulling image: rpc error: code = Unknown desc = Source image rejected: Required email wrong@mail.com not found (got []string{"sgrunert@redhat.com"})
```

你还可以对未签名的镜像进行策略测试。为此，你需要将键 `quay.io/crio/signed`
修改为类似 `quay.io/crio/unsigned`：

```console
> sed -i 's;quay.io/crio/signed;quay.io/crio/unsigned;' policy.json
```

如果你现在拉取容器镜像，CRI-O 将报错此镜像不存在签名：

```console
> sudo crictl pull quay.io/crio/unsigned
FATA[…] pulling image: rpc error: code = Unknown desc = SignatureValidationFailed: Source image rejected: A signature was required, but no signature exists
```

需要强调的是，CRI-O 将签名中的 `.critical.identity.docker-reference` 字段与镜像仓库进行匹配。
例如，如果你要验证镜像 `registry.k8s.io/kube-apiserver-amd64:v1.28.0-alpha.3`，
则相应的 `docker-reference` 须是 `registry.k8s.io/kube-apiserver-amd64`：

```console
> cosign verify registry.k8s.io/kube-apiserver-amd64:v1.28.0-alpha.3 \
    --certificate-identity krel-trust@k8s-releng-prod.iam.gserviceaccount.com \
    --certificate-oidc-issuer https://accounts.google.com \
    | jq -r '.[0].critical.identity."docker-reference"'
…

registry.k8s.io/kubernetes/kube-apiserver-amd64
```

Kubernetes 社区引入了 `registry.k8s.io` 作为各种镜像仓库的代理镜像。
在 [kpromo v4.0.2][kpromo] 版本发布之前，镜像已使用实际镜像签名而不是使用
`registry.k8s.io` 签名：

[kpromo]: https://github.com/kubernetes-sigs/promo-tools/releases/tag/v4.0.2

```console
> cosign verify registry.k8s.io/kube-apiserver-amd64:v1.28.0-alpha.2 \
    --certificate-identity krel-trust@k8s-releng-prod.iam.gserviceaccount.com \
    --certificate-oidc-issuer https://accounts.google.com \
    | jq -r '.[0].critical.identity."docker-reference"'
…

asia-northeast2-docker.pkg.dev/k8s-artifacts-prod/images/kubernetes/kube-apiserver-amd64
```

将 `docker-reference` 更改为 `registry.k8s.io` 使最终用户更容易验证签名，
因为他们不需要知道所使用的底层基础设施的详细信息。设置镜像签名身份的特性已通过
`sign --sign-container-identity` 标志添加到 `cosign`，并将成为即将发布的版本的一部分。

[cosign-pr]: https://github.com/sigstore/cosign/pull/2984

[最近在 Kubernetes 中添加了][pr-117717]镜像拉取错误码 `SignatureValidationFailed`，
将从 v1.28 版本开始可用。这个错误码允许最终用户直接从 kubectl CLI 了解镜像拉取失败的原因。
例如，如果你使用要求对 `quay.io/crio/unsigned` 进行签名的策略同时运行 CRI-O 和 Kubernetes，
那么 Pod 的定义如下：

[pr-117717]: https://github.com/kubernetes/kubernetes/pull/117717

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
    - name: container
      image: quay.io/crio/unsigned
```

将在应用 Pod 清单时造成 `SignatureValidationFailed` 错误：

```console
> kubectl apply -f pod.yaml
pod/pod created
```

```console
> kubectl get pods
NAME   READY   STATUS                      RESTARTS   AGE
pod    0/1     SignatureValidationFailed   0          4s
```

```console
> kubectl describe pod pod | tail -n8
  Type     Reason     Age                From               Message
  ----     ------     ----               ----               -------
  Normal   Scheduled  58s                default-scheduler  Successfully assigned default/pod to 127.0.0.1
  Normal   BackOff    22s (x2 over 55s)  kubelet            Back-off pulling image "quay.io/crio/unsigned"
  Warning  Failed     22s (x2 over 55s)  kubelet            Error: ImagePullBackOff
  Normal   Pulling    9s (x3 over 58s)   kubelet            Pulling image "quay.io/crio/unsigned"
  Warning  Failed     6s (x3 over 55s)   kubelet            Failed to pull image "quay.io/crio/unsigned": SignatureValidationFailed: Source image rejected: A signature was required, but no signature exists
  Warning  Failed     6s (x3 over 55s)   kubelet            Error: SignatureValidationFailed
```

这种整体行为提供了更符合 Kubernetes 原生体验的方式，并且不依赖于在集群中安装的第三方软件。

还有一些特殊情况需要考虑：例如，如果你希望像策略控制器那样允许按命名空间设置策略，怎么办？
好消息是，CRI-O 在 v1.28 版本中即将推出这个特性！CRI-O 将支持 `--signature-policy-dir` /
`signature_policy_dir` 选项，为命名空间隔离的签名策略的 Pod 定义根路径。
这意味着 CRI-O 将查找该路径，并组装一个类似 `<SIGNATURE_POLICY_DIR>/<NAMESPACE>.json`
的策略，在镜像拉取时如果存在则使用该策略。如果（[通过沙盒配置][sandbox-config]）
在镜像拉取时未提供 Pod 命名空间，或者串接的路径不存在，则 CRI-O 的全局策略将用作后备。

[sandbox-config]: https://github.com/kubernetes/cri-api/blob/e5515a5/pkg/apis/runtime/v1/api.proto#L1448

另一个需要考虑的特殊情况对于容器运行时中正确的签名验证至关重要：kubelet
仅在磁盘上不存在镜像时才调用容器镜像拉取。这意味着来自 Kubernetes 命名空间 A
的不受限策略可以允许拉取一个镜像，而命名空间 B 则无法强制执行该策略，
因为它已经存在于节点上了。最后，CRI-O 必须在容器创建时验证策略，而不仅仅是在镜像拉取时。
这一事实使情况变得更加复杂，因为 CRI 在容器创建时并没有真正传递用户指定的镜像引用，
而是传递已经解析过的镜像 ID 或摘要。[对 CRI 进行小改动][pr-118652] 有助于解决这个问题。

[pr-118652]: https://github.com/kubernetes/kubernetes/pull/118652

现在一切都发生在容器运行时中，大家必须维护和定义策略以提供良好的用户体验。
策略控制器的 CRD 非常出色，我们可以想象集群中的一个守护进程可以按命名空间为 CRI-O 编写策略。
这将使任何额外的回调过时，并将验证镜像签名的责任移交给实际拉取镜像的实例。
我已经评估了在纯 Kubernetes 中实现更好的容器镜像签名验证的其他可能途径，
但我没有找到一个很好的原生 API 解决方案。这意味着我相信 CRD 是正确的方法，
但用户仍然需要一个实际有作用的实例。

[thread]: https://groups.google.com/g/kubernetes-sig-node/c/kgpxqcsJ7Vc/m/7X7t_ElsAgAJ

感谢阅读这篇博文！如果你对更多内容感兴趣，想提供反馈或寻求帮助，请随时通过
[Slack (#crio)][slack] 或 [SIG Node 邮件列表][mail]直接联系我。

[slack]: https://kubernetes.slack.com/messages/crio
[mail]: https://groups.google.com/forum/#!forum/kubernetes-sig-node
