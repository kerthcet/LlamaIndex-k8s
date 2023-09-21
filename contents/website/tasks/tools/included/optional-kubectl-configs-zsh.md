---
title: "zsh 自动补全"
description: "zsh 自动补全的一些可选配置"
headless: true
_build:
  list: never
  render: never
  publishResources: false
---

kubectl 通过命令 `kubectl completion zsh` 生成 Zsh 自动补全脚本。
在 Shell 中导入（Sourcing）该自动补全脚本，将启动 kubectl 自动补全功能。

为了在所有的 Shell 会话中实现此功能，请将下面内容加入到文件 `~/.zshrc` 中。

```zsh
source <(kubectl completion zsh)
```

如果你为 kubectl 定义了别名，kubectl 自动补全将自动使用它。

重新加载 Shell 后，kubectl 自动补全功能将立即生效。

如果你收到 `2: command not found: compdef` 这样的错误提示，那请将下面内容添加到
`~/.zshrc` 文件的开头：

```zsh
autoload -Uz compinit
compinit
```
