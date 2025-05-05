<p align="center">
  <img alt="Morphik Logo" src="assets/morphik_logo.png">
</p>
<p align="center">
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/morphik-org/morphik-core"/>
  <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/morphik-org/morphik-core"/>
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/morphik">
  <a href="https://discord.gg/BwMtv3Zaju"><img alt="Discord" src="https://img.shields.io/discord/1336524712817332276?logo=discord&label=discord"></a>
</p>

<!-- add a roadmap! - <a href="https://morphik.ai/roadmap">Roadmap</a> - -->
<!-- Add a changelog! - <a href="https://morphik.ai/changelog">Changelog</a> -->

<p align="center">
  <a href="https://docs.morphik.ai">Docs</a> - <a href="https://discord.gg/BwMtv3Zaju">Community</a> - <a href="https://docs.morphik.ai/blogs/gpt-vs-morphik-multimodal">Why Morphik?</a> - <a href="https://github.com/morphik-org/morphik-core/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

## Morphik is an alternative to traditional RAG for highly technical and visual documents.

[Morphik](https://morphik.ai) provides developers the tools to ingest, search (deep and shallow), transform, and manage unstructured and multimodal documents. Some of our features include:

- [Multimodal Search](https://docs.morphik.ai/concepts/colpali): We employ techniques such as ColPali to build search that actually *understands* the visual content of documents you provide. Search over images, PDFs, videos, and more with a single endpoint.
- [Knowledge Graphs](https://docs.morphik.ai/concepts/knowledge-graphs): Build knowledge graphs for domain-specific use cases in a single line of code. Use our battle-tested system prompts, or use your own.
- [Fast and Scalable Metadata Extraction](https://docs.morphik.ai/concepts/rules-processing): Extract metadata from documents - including bounding boxes, labeling, classification, and more.
- [Integrations](https://docs.morphik.ai/integrations): Integrate with existing tools and workflows. Including (but not limited to) Google Suite, Slack, and Confluence.
- [Cache-Augmented-Generation](https://docs.morphik.ai/python-sdk/create_cache): Create persistent KV-caches of your documents to speed up generation.

The best part? Morphik has a [free tier](https://www.morphik.ai/pricing) and is open source! Get started by signing up at [Morphik](https://www.morphik.ai/signup).

## Table of Contents
- [Getting Started with Morphik](#getting-started-with-morphik-recommended)
- [Self-hosting the open-source version](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [Open source vs paid](#open-source-vs-paid)

## Getting Started with Morphik (Recommended)

The fastest and easiest way to get started with Morphik is by signing up for free at [Morphik](https://www.morphik.ai/signup). Your first 200 pages and 100 queries are on us! After this, you can pay based on usage with discounted rates for heavier use.

## Self-hosting the open-source version

If you'd like to self-host Morphik, you can find the dedicated instruction [here](https://docs.morphik.ai/getting-started). We offer options for direct installation and installation via docker.

**Important**: Due to limited resources, we cannot provide full support for open-source deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

Once you've signed up for Morphik, you can get started with ingesting and search your data right away.


### Code (Example: Python SDK)
For programmers, we offer a [Python SDK](https://docs.morphik.ai/python-sdk/morphik) and a [REST API](https://docs.morphik.ai/api-reference/health-check). Ingesting a file is as simple as:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Similarly, searching and querying your data is easy too:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

You can also interact with Morphik via the Morphik Console. This is a web-based interface that allows you to ingest, search, and query your data. You can upload files, connect to different data sources, and chat with your data all within the same place.

### Model Context Protocol

Finally, you can also access Morphik via MCP. Instructions are available [here](https://docs.morphik.ai/using-morphik/mcp).


## Contributing
You're welcome to contribute to the project! We love:
- Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
- Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
- Pull requests

Currently, we're focused on improving speed, integrating with more tools, and finding the research papers that provide the most value to our users. If you have thoughts, let us know in the discord or in GitHub!

## Open source vs paid

Certain features - such as Morphik Console - are not available in the open-source version. Any feature in the `ee` namespace is not available in the open-source version and carries a different license. Any feature outside that is open source under the MIT expat license.

## Contributors

Visit our special thanks page dedicated to our contributors [here](https://docs.morphik.ai/special-thanks).

## PS
We took inspiration from [PostHog](https://posthog.com) while writing this README. If you're from PostHog, thank you ❤️
