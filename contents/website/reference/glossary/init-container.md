---
title: Init 容器（Init Container）
id: init-container
date: 2018-04-12
full_link: 
short_description: >
  应用容器运行前必须先运行完成的一个或多个 Init 容器（Init Container）。 

aka: 
tags:
- fundamental
---


应用{{< glossary_tooltip text="容器" term_id="container" >}}运行前必须先运行完成的一个或多个 Init 容器（Init Container）。


Init 容器像常规应用容器一样，只有一点不同：Init 容器必须在应用容器启动前运行完成。
Init 容器的运行顺序：一个 Init 容器必须在下一个 Init 容器开始前运行完成。
