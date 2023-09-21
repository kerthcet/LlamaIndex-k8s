---
title: 为上游 Kubernetes 代码库做出贡献
content_type: task
weight: 20
---


此页面描述如何为上游 `kubernetes/kubernetes` 项目做出贡献，如修复 Kubernetes API
文档或 Kubernetes 组件（例如 `kubeadm`、`kube-apiserver`、`kube-controller-manager` 等）
中发现的错误。

如果你仅想从上游代码重新生成 Kubernetes API 或 `kube-*` 组件的参考文档。请参考以下说明：

- [生成 Kubernetes API 的参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubernetes-api/)
- [生成 Kubernetes 组件和工具的参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubernetes-components/)

## {{% heading "prerequisites" %}}

- 你需要安装以下工具：

  - [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
  - [Golang](https://golang.org/doc/install) 的 1.13 版本或更高
  - [Docker](https://docs.docker.com/engine/installation/)
  - [etcd](https://github.com/coreos/etcd/)
  - [make](https://www.gnu.org/software/make/)
  - [gcc compiler/linker](https://gcc.gnu.org/)

- 你必须设置 `GOPATH` 环境变量，并且 `etcd` 的位置必须在 `PATH` 环境变量中。

- 你需要知道如何创建对 GitHub 代码仓库的拉取请求（Pull Request）。
  通常，这涉及创建代码仓库的派生副本。
  要获取更多的信息请参考[创建 PR](https://help.github.com/articles/creating-a-pull-request/) 和
  [GitHub 标准派生和 PR 工作流程](https://gist.github.com/Chaser324/ce0505fbed06b947d962)。


## 基本说明

Kubernetes API 和 `kube-*` 组件（例如 `kube-apiserver`、`kube-controller-manager`）的参考文档
是根据[上游 Kubernetes](https://github.com/kubernetes/kubernetes/) 中的源代码自动生成的。

当你在生成的文档中看到错误时，你可能需要考虑创建一个 PR 用来在上游项目中对其进行修复。

## 克隆 Kubernetes 代码仓库

如果你还没有 kubernetes/kubernetes 代码仓库，请参照下列命令获取：

```shell
mkdir $GOPATH/src
cd $GOPATH/src
go get github.com/kubernetes/kubernetes
```

确定你的 [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) 代码仓库克隆的根目录。
例如，如果按照前面的步骤获取代码仓库，则你的根目录为 `$GOPATH/src/github.com/kubernetes/kubernetes`。
接下来其余步骤将你的根目录称为 `<k8s-base>`。

确定你的 [kubernetes-sigs/reference-docs](https://github.com/kubernetes-sigs/reference-docs)
代码仓库克隆的根目录。
例如，如果按照前面的步骤获取代码仓库，则你的根目录为
`$GOPATH/src/github.com/kubernetes-sigs/reference-docs`。
接下来其余步骤将你的根目录称为 `<rdocs-base>`。

## 编辑 Kubernetes 源代码

Kubernetes API 参考文档是根据 OpenAPI 规范自动生成的，该规范是从 Kubernetes 源代码生成的。
如果要更改 API 参考文档，第一步是更改 Kubernetes 源代码中的一个或多个注释。

`kube-*` 组件的文档也是从上游源代码生成的。你必须更改与要修复的组件相关的代码，才能修复生成的文档。

### 更改上游 Kubernetes 源代码

{{< note >}}
以下步骤仅作为示例，不是通用步骤，具体情况因环境而异。
{{< /note >}}

以下在 Kubernetes 源代码中编辑注释的示例。

在你本地的 kubernetes/kubernetes 代码仓库中，检出默认分支，并确保它是最新的：

```shell
cd <k8s-base>
git checkout master
git pull https://github.com/kubernetes/kubernetes master
```

假设默认分支中的下面源文件中包含拼写错误 "atmost"：

[kubernetes/kubernetes/staging/src/k8s.io/api/apps/v1/types.go](https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/api/apps/v1/types.go)

在你的本地环境中，打开 `types.go` 文件，然后将 "atmost" 更改为 "at most"。

以下命令验证你已经更改了文件：

```shell
git status
```

输出显示你在 master 分支上，`types.go` 源文件已被修改：

```shell
On branch master
...
    modified:   staging/src/k8s.io/api/apps/v1/types.go
```

### 提交已编辑的文件

运行 `git add` 和 `git commit` 命令提交到目前为止所做的更改。
在下一步中，你将进行第二次提交，将更改分成两个提交很重要。

### 生成 OpenAPI 规范和相关文件

进入 `<k8s-base>` 目录并运行以下脚本：

```shell
hack/update-generated-swagger-docs.sh
hack/update-openapi-spec.sh
hack/update-generated-protobuf.sh
```

运行 `git status` 命令查看生成的文件。

```none
On branch master
...
    modified:   api/openapi-spec/swagger.json
    modified:   api/openapi-spec/v3/apis__apps__v1_openapi.json
    modified:   pkg/generated/openapi/zz_generated.openapi.go
    modified:   staging/src/k8s.io/api/apps/v1/generated.proto
    modified:   staging/src/k8s.io/api/apps/v1/types_swagger_doc_generated.go
```

查看 `api/openapi-spec/swagger.json` 的内容，以确保拼写错误已经被修正。
例如，你可以运行 `git diff -a api/openapi-spec/swagger.json` 命令。
这很重要，因为 `swagger.json` 是文档生成过程中第二阶段的输入。

运行 `git add` 和 `git commit` 命令来提交你的更改。现在你有两个提交（commits）：
一种包含编辑的 `types.go` 文件，另一种包含生成的 OpenAPI 规范和相关文件。
将这两个提交分开独立。也就是说，不要 squash 你的提交。

将你的更改作为 [PR](https://help.github.com/articles/creating-a-pull-request/) 
提交到 [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) 代码仓库的 master 分支。
关注你的 PR，并根据需要回复 reviewer 的评论。继续关注你的 PR，直到 PR 被合并为止。

[PR 57758](https://github.com/kubernetes/kubernetes/pull/57758) 是修复 Kubernetes
源代码中的拼写错误的拉取请求的示例。

{{< note >}}
确定要更改的正确源文件可能很棘手。在前面的示例中，官方的源文件位于 `kubernetes/kubernetes`
代码仓库的 `staging` 目录中。但是根据你的情况，`staging` 目录可能不是找到官方源文件的地方。
如果需要帮助，请阅读
[kubernetes/kubernetes](https://github.com/kubernetes/kubernetes/tree/master/staging)
代码仓库和相关代码仓库
（例如 [kubernetes/apiserver](https://github.com/kubernetes/apiserver/blob/master/README.md)）
中的 `README` 文件。
{{< /note >}}

### 将你的提交 Cherrypick 到发布分支

在上一节中，你在 master 分支中编辑了一个文件，然后运行了脚本用来生成 OpenAPI 规范和相关文件。
然后用 PR 将你的更改提交到 kubernetes/kubernetes 代码仓库的 master 分支中。
现在，需要将你的更改反向移植到已经发布的分支。
例如，假设 master 分支被用来开发 Kubernetes {{< skew latestVersion >}} 版，
并且你想将更改反向移植到 release-{{< skew prevMinorVersion >}} 分支。

回想一下，你的 PR 有两个提交：一个用于编辑 `types.go`，一个用于由脚本生成的文件。
下一步是将你的第一次提交 cherrypick 到 release-{{< skew prevMinorVersion >}} 分支。
这样做的原因是仅 cherrypick 编辑了 types.go 的提交，
而不是具有脚本运行结果的提交。
有关说明，请参见[提出 Cherry Pick](https://git.k8s.io/community/contributors/devel/sig-release/cherry-picks.md)。

{{< note >}}
提出 Cherry Pick 要求你有权在 PR 中设置标签和里程碑。如果你没有这些权限，
则需要与可以为你设置标签和里程碑的人员合作。
{{< /note >}}

当你发起 PR 将你的一个提交 cherry pick 到 release-{{< skew prevMinorVersion >}} 分支中时，
下一步是在本地环境的 release-{{< skew prevMinorVersion >}} 分支中运行如下脚本。

```shell
hack/update-generated-swagger-docs.sh
hack/update-openapi-spec.sh
hack/update-generated-protobuf.sh
hack/update-api-reference-docs.sh
```

现在将提交添加到你的 Cherry-Pick PR 中，该 PR 中包含最新生成的 OpenAPI 规范和相关文件。
关注你的 PR，直到其合并到 release-{{< skew prevMinorVersion >}} 分支中为止。

此时，master 分支和 release-{{< skew prevMinorVersion >}}
分支都具有更新的 `types.go` 文件和一组生成的文件，
这些文件反映了对 `types.go` 所做的更改。
请注意，生成的 OpenAPI 规范和其他 release-{{< skew prevMinorVersion >}}
分支中生成的文件不一定与 master 分支中生成的文件相同。
release-{{< skew prevMinorVersion >}} 分支中生成的文件仅包含来自
Kubernetes {{< skew prevMinorVersion >}} 的 API 元素。
master 分支中生成的文件可能包含不在 {{< skew prevMinorVersion >}}
中但正在为 {{< skew latestVersion >}} 开发的 API 元素。

## 生成已发布的参考文档

上一节显示了如何编辑源文件然后生成多个文件，包括在 `kubernetes/kubernetes` 代码仓库中的
`api/openapi-spec/swagger.json`。`swagger.json` 文件是 OpenAPI 定义文件，可用于生成 API 参考文档。

现在，你可以按照
[生成 Kubernetes API 的参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubernetes-api/)
指南来生成
[已发布的 Kubernetes API 参考文档](/docs/reference/generated/kubernetes-api/{{< param "version" >}}/)。

## {{% heading "whatsnext" %}}

* [生成 Kubernetes API 的参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubernetes-api/)
* [为 Kubernetes 组件和工具生成参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubernetes-components/)
* [生成 kubectl 命令的参考文档](/zh-cn/docs/contribute/generate-ref-docs/kubectl/)
