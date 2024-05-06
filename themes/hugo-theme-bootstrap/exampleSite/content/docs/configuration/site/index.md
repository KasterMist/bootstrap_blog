+++
title = "Site Configuration"
date = 2021-11-27T19:53:24+08:00
featured = false
comment = true
toc = true
reward = true
pinned = false
categories = [
  "Configuration"
]
tags = [
]
series = [
  "Docs"
]
images = []
navWeight = 990
authors = ["RazonYang"]
+++

The site configuration is located in `config/_default/config.toml` by default.

<!--more-->

| Name | Type | Default | Description
|---|:-:|:-:|---
| `title` | String | - | Site title.
| `baseURL` | String | - | Site base URL.
| `copyright` | String | - | Site copyright. The `{year}` placeholder will be replaced with the current year.
| `defaultContentLanguage` | String | `en` | Default site's language.
| `defaultContentLanguageInSubdir` | Boolean | `false` |
| `paginate` | Integer | `10` |
| `paginatePath` | String | `page` |
| `enableRobotsTXT` | Boolean | `true` |
| `disqusShortname` | String | - | [Disqus]({{< ref "/docs/widgets/comments#disqus" >}}) shortname.
| `services` | Object | - |
| `services.googleAnalytics` | Object | - | Google Analytics, supports GA4 only.
| `services.googleAnalytics.ID` | String | - | Measurement ID.
| `author` | Object | - | [Author Widget]({{< ref "/docs/widgets/author" >}}).

See also [All Configuration Settings](https://gohugo.io/getting-started/configuration/#all-configuration-settings).
