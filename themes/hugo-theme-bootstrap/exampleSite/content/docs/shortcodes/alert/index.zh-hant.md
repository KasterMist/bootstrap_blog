+++
title = "Alert Shortcode"
linkTitle = "Alert"
date = "2020-10-22"
description = "A detailed description of Alert shortcode"
featured = false
categories = [
  "Shortcode"
]
tags = [
  "Alert"
]
series = [
  "文檔"
]
images = [
]
aliases = [
  "/en/posts/shortcodes/alert"
]
authors = ["RazonYang"]
+++

本文將介紹如何使用 `alert` shortcode。

<!--more-->

## 語法

### Inline

```markdown
{{</* alert [type] */>}}INLINE ALERT{{</* /alert */>}}
```

### Block

```markdown
{{</* alert [type] */>}}
BLOCK ALERT
{{</* /alert */>}}
```

> The parameter `type` is optional. Default to `info`, available values: `info`, `success`, `warning` and `danger`.

### 帶有標題

```markdown
{{</* alert [type] */>}}
{{</* alert-heading */>}}Well Done!{{</* /alert-heading */>}}
ALERT MESSAGE
{{</* /alert */>}}
```

### 帶有 Markdown 格式

````markdown
{{%/* alert warning */%}}
Alert Shortcode with Markdown Syntax:
```bash
$ echo 'An example of alert shortcode with the Markdown syntax'
```
{{%/* /alert */%}}
````

{{% alert warning %}}
{{% code/alert-with-markdown-example %}}
{{% /alert %}}

請注意，你需要開啟 `markup.goldmark.renderer.unsafe` 配置。

{{< code-toggle filename="config" >}}
{{% config/markup-goldmark-renderer-unsafe %}}
{{< /code-toggle >}}

## 例子

{{< alert >}}Info{{< /alert >}}

{{< alert success >}}
  Aww yeah, you successfully read this important alert message. This example text is going to run a bit longer so that you can see how spacing within an alert works with this kind of content.
{{< /alert >}}

{{< alert warning >}}Warning{{< /alert >}}

{{< alert danger >}}Danger{{< /alert >}}

{{< alert "success" >}}
  {{< alert-heading >}}Well Done!{{< /alert-heading >}}
  Aww yeah, you successfully read this important alert message. This example text is going to run a bit longer so that you can see how spacing within an alert works with this kind of content.
{{< /alert >}}
