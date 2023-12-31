---
title: Kubelet 配置 (v1alpha1)
content_type: tool-reference
package: kubelet.config.k8s.io/v1alpha1
---


## 资源类型

- [CredentialProviderConfig](#kubelet-config-k8s-io-v1alpha1-CredentialProviderConfig)

## `CredentialProviderConfig`     {#kubelet-config-k8s-io-v1alpha1-CredentialProviderConfig}

CredentialProviderConfig 包含有关每个 exec 凭据提供者的配置信息。
Kubelet 从磁盘上读取这些配置信息，并根据 CredentialProvider 类型启用各个提供者。

<table class="table">
<tbody>

<tr><td><code>apiVersion</code><br/>string</td><td><code>kubelet.config.k8s.io/v1alpha1</code></td></tr>
<tr><td><code>kind</code><br/>string</td><td><code>CredentialProviderConfig</code></td></tr>
<tr><td><code>providers</code> <B>[必需]</B><br/>
<a href="#kubelet-config-k8s-io-v1alpha1-CredentialProvider"><code>[]CredentialProvider</code></a>
</td>
<td>
  <code>providers</code> 是一组凭据提供者插件，这些插件会被 kubelet 启用。
多个提供者可以匹配到同一镜像上，这时，来自所有提供者的凭据信息都会返回给 kubelet。
如果针对同一镜像调用了多个提供者，则结果会被组合起来。如果提供者返回的认证主键有重复，
列表中先出现的提供者所返回的值将被使用。
</td>
</tr>
</tbody>
</table>

## `CredentialProvider`     {#kubelet-config-k8s-io-v1alpha1-CredentialProvider}

**出现在：**

- [CredentialProviderConfig](#kubelet-config-k8s-io-v1alpha1-CredentialProviderConfig)

CredentialProvider 代表的是要被 kubelet 调用的一个 exec 插件。
这一插件只会在所拉取的镜像与该插件所处理的镜像匹配时才会被调用（参见 <code>matchImages</code>）。

<table class="table">
<tbody>

<tr><td><code>name</code> <B>[必需]</B><br/>
<code>string</code>
</td>
<td>
  <code>name</code> 是凭据提供者的名称（必需）。此名称必须与 kubelet
  所看到的提供者可执行文件的名称匹配。可执行文件必须位于 kubelet 的
  <code>bin</code> 目录（通过 <code>--image-credential-provider-bin-dir</code> 设置）下。
</td>
</tr>

<tr><td><code>matchImages</code> <B>[必需]</B><br/>
<code>[]string</code>
</td>
<td>
<p><code>matchImages</code> 是一个必须设置的字符串列表，用来匹配镜像以便确定是否要调用此提供者。
如果字符串之一与 kubelet 所请求的镜像匹配，则此插件会被调用并给予提供凭证的机会。
镜像应该包含镜像库域名和 URL 路径。</p>
<p><code>matchImages</code> 中的每个条目都是一个模式字符串，其中可以包含端口号和路径。
域名部分可以包含统配符，但端口或路径部分不可以。通配符可以用作子域名，例如
<code>*.k8s.io</code> 或 <code>k8s.*.io</code>，以及顶级域名，如 <code>k8s.*</code>。</p>
<p>对类似 <code>app*.k8s.io</code> 这类部分子域名的匹配也是支持的。
每个通配符只能用来匹配一个子域名段，所以 <code>*.io</code> 不会匹配 <code>*.k8s.io</code>。</p>
<p>镜像与 <code>matchImages</code> 之间存在匹配时，以下条件都要满足：</p>
<ul>
  <li>二者均包含相同个数的域名部分，并且每个域名部分都对应匹配；</li>
  <li><code>matchImages</code> 条目中的 URL 路径部分必须是目标镜像的 URL 路径的前缀；</li>
  <li>如果 <code>matchImages</code> 条目中包含端口号，则端口号也必须与镜像端口号匹配。</li>
</ul>
<p><code>matchImages</code> 的一些示例如下：</p>
<ul>
<li><code>123456789.dkr.ecr.us-east-1.amazonaws.com</code></li>
<li><code>*.azurecr.io</code></li>
<li><code>gcr.io</code></li>
<li><code>*.*.registry.io</code></li>
<li><code>registry.io:8080/path</code></li>
</ul>
</td>
</tr>

<tr><td><code>defaultCacheDuration</code> <B>[必需]</B><br/>
<a href="https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1#Duration"><code>meta/v1.Duration</code></a>
</td>
<td>
   <code>defaultCacheDuration</code> 是插件在内存中缓存凭据的默认时长，
在插件响应中没有给出缓存时长时，使用这里设置的值。此字段是必需的。
</td>
</tr>

<tr><td><code>apiVersion</code> <B>[必需]</B><br/>
<code>string</code>
</td>
<td>
  <p>要求 exec 插件 CredentialProviderRequest 请求的输入版本。
  所返回的 CredentialProviderResponse 必须使用与输入相同的编码版本。当前支持的值有：</p>
  <ul>
    <li><code>credentialprovider.kubelet.k8s.io/v1alpha1</code></li>
  </ul>
</td>
</tr>

<tr><td><code>args</code><br/>
<code>[]string</code>
</td>
<td>
  在执行插件可执行文件时要传递给命令的参数。
</td>
</tr>

<tr><td><code>env</code><br/>
<a href="#kubelet-config-k8s-io-v1alpha1-ExecEnvVar"><code>[]ExecEnvVar</code></a>
</td>
<td>
  <code>env</code> 定义要提供给插件进程的额外的环境变量。
这些环境变量会与主机上的其他环境变量以及 client-go 所使用的环境变量组合起来，
一起传递给插件。
</td>
</tr>
</tbody>
</table>

## `ExecEnvVar`     {#kubelet-config-k8s-io-v1alpha1-ExecEnvVar}

**出现在：**

- [CredentialProvider](#kubelet-config-k8s-io-v1alpha1-CredentialProvider)

ExecEnvVar 用来在执行基于 exec 的凭据插件时设置环境变量。

<table class="table">
<tbody>

<tr><td><code>name</code> <B>[必需]</B><br/>
<code>string</code>
</td>
<td>
  环境变量名称。
</td>
</tr>

<tr><td><code>value</code> <B>[必需]</B><br/>
<code>string</code>
</td>
<td>
  环境变量取值。
</td>
</tr>
</tbody>
</table>

