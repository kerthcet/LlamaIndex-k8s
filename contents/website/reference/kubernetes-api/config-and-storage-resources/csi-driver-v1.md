---
api_metadata:
  apiVersion: "storage.k8s.io/v1"
  import: "k8s.io/api/storage/v1"
  kind: "CSIDriver"
content_type: "api_reference"
description: "CSIDriver 抓取集群上部署的容器存储接口（CSI）卷驱动有关的信息。"
title: "CSIDriver"
weight: 8
---

`apiVersion: storage.k8s.io/v1`

`import "k8s.io/api/storage/v1"`

## CSIDriver {#CSIDriver}

CSIDriver 抓取集群上部署的容器存储接口（CSI）卷驱动有关的信息。
Kubernetes 挂接/解除挂接控制器使用此对象来决定是否需要挂接。
Kubelet 使用此对象决定挂载时是否需要传递 Pod 信息。
CSIDriver 对象未划分命名空间。

<hr>

- **apiVersion**: storage.k8s.io/v1

- **kind**: CSIDriver

- **metadata** (<a href="{{< ref "../common-definitions/object-meta#ObjectMeta" >}}">ObjectMeta</a>)
  
  标准的对象元数据。
  `metadata.name` 表示此对象引用的 CSI 驱动的名称；
  它必须与该驱动的 CSI GetPluginName() 调用返回的名称相同。
  驱动名称不得超过 63 个字符，以字母、数字（[a-z0-9A-Z]）开头和结尾，
  中间可包含短划线（-）、英文句点（.）、字母和数字。
  更多信息：
  https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata

- **spec** (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriverSpec" >}}">CSIDriverSpec</a>)，必需
  
  spec 表示 CSI 驱动的规约。

## CSIDriverSpec {#CSIDriverSpec}

CSIDriverSpec 是 CSIDriver 的规约。

<hr>

- **attachRequired** (boolean)
  
  attachRequired 表示这个 CSI 卷驱动需要挂接操作
  （因为它实现了 CSI ControllerPublishVolume() 方法），
  Kubernetes 挂接/解除挂接控制器应调用挂接卷接口，
  以检查卷挂接（volumeattachment）状态并在继续挂载之前等待卷被挂接。
  CSI 外部挂接器与 CSI 卷驱动配合使用，并在挂接操作完成时更新 volumeattachment 状态。
  如果 CSIDriverRegistry 特性门控被启用且此值指定为 false，将跳过挂接操作。
  否则将调用挂接操作。
  
  此字段不可变更。

- **fsGroupPolicy** (string)
  
  fsGroupPolicy 定义底层卷是否支持在挂载之前更改卷的所有权和权限。
  有关更多详细信息，请参考特定的 FSGroupPolicy 值。
  
  此字段不可变更。
  
  默认为 ReadWriteOnceWithFSType，这会检查每个卷，以决定 Kubernetes 是否应修改卷的所有权和权限。
  采用默认策略时，如果定义了 fstype 且卷的访问模式包含 ReadWriteOnce，将仅应用定义的 fsGroup。

- **podInfoOnMount** (boolean)

  如果 podInfoOnMount 设为 true，则表示在挂载操作期间这个 CSI 卷驱动需要更多的
  Pod 信息（例如 podName 和 podUID 等）。
  如果设为 false，则挂载时将不传递 Pod 信息。默认为 false。
  
  CSI 驱动将 podInfoOnMount 指定为驱动部署的一部分。
  如果为 true，Kubelet 将在 CSI NodePublishVolume() 调用中作为 VolumeContext 传递 Pod 信息。
  CSI 驱动负责解析和校验作为 VolumeContext 传递进来的信息。

  如果 podInfoOnMount 设为 true，将传递以下 VolumeConext。
  此列表可能变大，但将使用前缀。

  - "csi.storage.k8s.io/pod.name": pod.name
  - "csi.storage.k8s.io/pod.namespace": pod.namespace
  - "csi.storage.k8s.io/pod.uid": string(pod.UID)
  - "csi.storage.k8s.io/ephemeral":
    如果此卷是 CSIVolumeSource 定义的一个临时内联卷，则为 “true”，否则为 “false”

  “csi.storage.k8s.io/ephemeral” 是 Kubernetes 1.16 中一个新的功能特性。
  只有同时支持 “Persistent” 和 “Ephemeral” VolumeLifecycleMode 的驱动，此字段才是必需的。
  其他驱动可以保持禁用 Pod 信息或忽略此字段。
  由于 Kubernetes 1.15 不支持此字段，所以在这类集群上部署驱动时，只能支持一种模式。
  该部署就决定了是哪种模式，例如通过驱动的命令行参数。
  
  此字段不可变更。

- **requiresRepublish** (boolean)
  
  requiresRepublish 表示 CSI 驱动想要 `NodePublishVolume` 被周期性地调用，
  以反映已挂载卷中的任何可能的变化。
  此字段默认为 false。
  
  注：成功完成对 NodePublishVolume 的初始调用后，对 NodePublishVolume 的后续调用只应更新卷的内容。
  新的挂载点将不会被运行的容器察觉。

- **seLinuxMount** (boolean)

  seLinuxMount 指定 CSI 驱动是否支持 "-o context" 挂载选项。

  当值为 “true” 时，CSI 驱动必须确保该 CSI 驱动提供的所有卷可以分别用不同的 `-o context` 选项进行挂载。
  这对于将卷作为块设备上的文件系统或作为独立共享卷提供的存储后端来说是典型的方法。
  当 Kubernetes 挂载在 Pod 中使用的已显式设置 SELinux 上下文的 ReadWriteOncePod 卷时，
  将使用 "-o context=xyz" 挂载选项调用 NodeStage / NodePublish。
  未来可能会扩展到其他的卷访问模式（AccessModes）。在任何情况下，Kubernetes 都会确保该卷仅使用同一 SELinux 上下文进行挂载。

  当值为 “false” 时，Kubernetes 不会将任何特殊的 SELinux 挂载选项传递给驱动。
  这通常用于代表更大共享文件系统的子目录的卷。
  
  默认为 “false”。

- **storageCapacity** (boolean)
  
  如果设为 true，则 storageCapacity 表示 CSI 卷驱动希望 Pod 调度时考虑存储容量，
  驱动部署将通过创建包含容量信息的 CSIStorageCapacity 对象来报告该存储容量。
  
  部署驱动时可以立即启用这个检查。
  这种情况下，只有此驱动部署已发布某些合适的 CSIStorageCapacity 对象，
  才会继续制备新的卷，然后进行绑定。
  
  换言之，可以在未设置此字段或此字段为 false 的情况下部署驱动，
  并且可以在发布存储容量信息后再修改此字段。
  
  此字段在 Kubernetes 1.22 及更早版本中不可变更，但现在可以变更。

- **tokenRequests** ([]TokenRequest)
  
  **原子性：将在合并期间被替换**
  
  tokenRequests 表示 CSI 驱动需要供挂载卷所用的 Pod 的服务帐户令牌，进行必要的鉴权。
  Kubelet 将在 CSI NodePublishVolume 调用中传递 VolumeContext 中的令牌。
  CSI 驱动应解析和校验以下 VolumeContext：

  ```
  "csi.storage.k8s.io/serviceAccount.tokens": {
    "<audience>": {
      "token": <token>,
      "expirationTimestamp": <expiration timestamp in RFC3339>,
    },
    ...
  }
  ```

  注：每个 tokenRequest 中的受众应该不同，且最多有一个令牌是空字符串。
  要在令牌过期后接收一个新的令牌，requiresRepublish 可用于周期性地触发 NodePublishVolume。
  
  <a name="TokenRequest"></a>
  **tokenRequest 包含一个服务帐户令牌的参数。**

  - **tokenRequests.audience** (string)，必需
    
    audience 是 “TokenRequestSpec” 中令牌的目标受众。
    它默认为 kube apiserver 的受众。
  
  - **tokenRequests.expirationSeconds** (int64)
    
    expirationSeconds 是 “TokenRequestSpec” 中令牌的有效期。
    它具有与 “TokenRequestSpec” 中 “expirationSeconds” 相同的默认值。

- **volumeLifecycleModes** ([]string)
  
  **集合：唯一值将在合并期间被保留**
  
  volumeLifecycleModes 定义这个 CSI 卷驱动支持哪种类别的卷。
  如果列表为空，则默认值为 “Persistent”，这是 CSI 规范定义的用法，
  并通过常用的 PV/PVC 机制在 Kubernetes 中实现。

  另一种模式是 “Ephemeral”。
  在这种模式下，在 Pod 规约中用 CSIVolumeSource 以内联方式定义卷，其生命周期与该 Pod 的生命周期相关联。
  驱动必须感知到这一点，因为只有针对这种卷才会接收到 NodePublishVolume 调用。

  有关实现此模式的更多信息，请参阅
  https://kubernetes-csi.github.io/docs/ephemeral-local-volumes.html。
  驱动可以支持其中一种或多种模式，将来可能会添加更多模式。
  
  此字段处于 Beta 阶段。此字段不可变更。

## CSIDriverList {#CSIDriverList}

CSIDriverList 是 CSIDriver 对象的集合。

<hr>

- **apiVersion**: storage.k8s.io/v1

- **kind**: CSIDriverList

- **metadata** (<a href="{{< ref "../common-definitions/list-meta#ListMeta" >}}">ListMeta</a>)
  
  标准的列表元数据。更多信息：
  https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata

- **items** ([]<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>)，必需
  
  items 是 CSIDriver 的列表。

## 操作 {#Operations}

<hr>

### `get` 读取指定的 CSIDriver

#### HTTP 请求

GET /apis/storage.k8s.io/v1/csidrivers/{name}

#### 参数

- **name** (**路径参数**): string，必需
  
  CSIDriver 的名称。

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): OK

401: Unauthorized

### `list` 列出或观测类别为 CSIDriver 的对象

#### HTTP 请求

GET /apis/storage.k8s.io/v1/csidrivers

#### 参数

- **allowWatchBookmarks** (**查询参数**): boolean
  
  <a href="{{< ref "../common-parameters/common-parameters#allowWatchBookmarks" >}}">allowWatchBookmarks</a>

- **continue** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#continue" >}}">continue</a>

- **fieldSelector** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldSelector" >}}">fieldSelector</a>

- **labelSelector** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#labelSelector" >}}">labelSelector</a>

- **limit** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#limit" >}}">limit</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **resourceVersion** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#resourceVersion" >}}">resourceVersion</a>

- **resourceVersionMatch** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#resourceVersionMatch" >}}">resourceVersionMatch</a>

- **sendInitialEvents** (**查询参数**): boolean

  <a href="{{< ref "../common-parameters/common-parameters#sendInitialEvents" >}}">sendInitialEvents</a>

- **timeoutSeconds** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#timeoutSeconds" >}}">timeoutSeconds</a>

- **watch** (**查询参数**): boolean
  
  <a href="{{< ref "../common-parameters/common-parameters#watch" >}}">watch</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriverList" >}}">CSIDriverList</a>): OK

401: Unauthorized

### `create` 创建 CSIDriver

#### HTTP 请求

POST /apis/storage.k8s.io/v1/csidrivers

#### 参数

- **body**: <a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>，必需

- **dryRun** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): OK

201 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): Created

202 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): Accepted

401: Unauthorized

### `update` 替换指定的 CSIDriver

#### HTTP 请求

PUT /apis/storage.k8s.io/v1/csidrivers/{name}

#### 参数

- **name** (**路径参数**): string，必需
  
  CSIDriver 的名称。

- **body**: <a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>，必需

- **dryRun** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): OK

201 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): Created

401: Unauthorized

### `patch` 部分更新指定的 CSIDriver

#### HTTP 请求

PATCH /apis/storage.k8s.io/v1/csidrivers/{name}

#### 参数

- **name** (**路径参数**): string，必需
  
  CSIDriver 的名称。

- **body**: <a href="{{< ref "../common-definitions/patch#Patch" >}}">Patch</a>，必需

- **dryRun** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldManager** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldManager" >}}">fieldManager</a>

- **fieldValidation** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldValidation" >}}">fieldValidation</a>

- **force** (**查询参数**): boolean
  
  <a href="{{< ref "../common-parameters/common-parameters#force" >}}">force</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): OK

201 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): Created

401: Unauthorized

### `delete` 删除 CSIDriver

#### HTTP 请求

DELETE /apis/storage.k8s.io/v1/csidrivers/{name}

#### 参数

- **name** (**路径参数**): string，必需
  
  CSIDriver 的名称。

- **body**: <a href="{{< ref "../common-definitions/delete-options#DeleteOptions" >}}">DeleteOptions</a>

- **dryRun** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **gracePeriodSeconds** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#gracePeriodSeconds" >}}">gracePeriodSeconds</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **propagationPolicy** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#propagationPolicy" >}}">propagationPolicy</a>

#### 响应

200 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): OK

202 (<a href="{{< ref "../config-and-storage-resources/csi-driver-v1#CSIDriver" >}}">CSIDriver</a>): Accepted

401: Unauthorized

### `deletecollection` 删除 CSIDriver 的集合

#### HTTP 请求

DELETE /apis/storage.k8s.io/v1/csidrivers

#### 参数

- **body**: <a href="{{< ref "../common-definitions/delete-options#DeleteOptions" >}}">DeleteOptions</a>

- **continue** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#continue" >}}">continue</a>

- **dryRun** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#dryRun" >}}">dryRun</a>

- **fieldSelector** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#fieldSelector" >}}">fieldSelector</a>

- **gracePeriodSeconds** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#gracePeriodSeconds" >}}">gracePeriodSeconds</a>

- **labelSelector** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#labelSelector" >}}">labelSelector</a>

- **limit** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#limit" >}}">limit</a>

- **pretty** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#pretty" >}}">pretty</a>

- **propagationPolicy** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#propagationPolicy" >}}">propagationPolicy</a>

- **resourceVersion** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#resourceVersion" >}}">resourceVersion</a>

- **resourceVersionMatch** (**查询参数**): string
  
  <a href="{{< ref "../common-parameters/common-parameters#resourceVersionMatch" >}}">resourceVersionMatch</a>

- **sendInitialEvents** (**查询参数**): boolean

  <a href="{{< ref "../common-parameters/common-parameters#sendInitialEvents" >}}">sendInitialEvents</a>

- **timeoutSeconds** (**查询参数**): integer
  
  <a href="{{< ref "../common-parameters/common-parameters#timeoutSeconds" >}}">timeoutSeconds</a>

#### 响应

200 (<a href="{{< ref "../common-definitions/status#Status" >}}">Status</a>): OK

401: Unauthorized
