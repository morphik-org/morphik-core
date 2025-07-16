# Changelog

## 0.1.0-alpha.1 (2025-07-16)

Full Changelog: [v0.0.1-alpha.0...v0.1.0-alpha.1](https://github.com/morphik-org/morphik-core/compare/v0.0.1-alpha.0...v0.1.0-alpha.1)

### Features

* add 30 second statement timeout to Postgres connection settings ([536ef96](https://github.com/morphik-org/morphik-core/commit/536ef9623948c312c83f9748e3b45d9be482c63b))
* add chat title update functionality and enhance chat sidebar with search and edit capabilities ([ee578e5](https://github.com/morphik-org/morphik-core/commit/ee578e5d297df10a0a174ecc19577d2be704b2cc))
* add convert_to_markdown and ingest_output actions with UI integration for workflow processing ([6281ec3](https://github.com/morphik-org/morphik-core/commit/6281ec3a0c5acffbd13224584ae382910bfb79db))
* add display_mode option to control image cropping in agent responses ([90d9065](https://github.com/morphik-org/morphik-core/commit/90d9065cb81afa3f6725675942fda42981c10df9))
* add padding parameter to retrieve additional context around matched chunks ([bc047c7](https://github.com/morphik-org/morphik-core/commit/bc047c7e7ef436f53ec09676bded48d8c3ae4409))
* add request-level CPU profiling with yappi middleware ([eddb412](https://github.com/morphik-org/morphik-core/commit/eddb412f16b7bd26d8943937f70c0857a20de429))
* add upload button to empty state and optimize vector normalization ([e6c6711](https://github.com/morphik-org/morphik-core/commit/e6c67110cf5fadb0638245facf079bad7c04647f))


### Bug Fixes

* convert query embedding to float tensor for ColQwen2.5 processor compatibility ([090124b](https://github.com/morphik-org/morphik-core/commit/090124b11ef23cd57ae746184510267ebaca6a62))
* no header in case of zero pdf ([#225](https://github.com/morphik-org/morphik-core/issues/225)) ([dc59b97](https://github.com/morphik-org/morphik-core/commit/dc59b976c3d90c599d85ee7f713478ba66b1bce3))
* remove isoformat conversion when getting current UTC time ([3585395](https://github.com/morphik-org/morphik-core/commit/3585395dc215327d826b5c9d66c1c165825684f7))
* Resolve llama-cpp-python installation error ([#224](https://github.com/morphik-org/morphik-core/issues/224)) ([061508e](https://github.com/morphik-org/morphik-core/commit/061508e85af64004d16b82d9a829dd2a5694d3bb))
* sanitize extracted text by removing null bytes and control characters for PostgreSQL compatibility ([02829cc](https://github.com/morphik-org/morphik-core/commit/02829cc1f1683c2eef11a4704474716c42844485))
* ui component build failure ([ce82b74](https://github.com/morphik-org/morphik-core/commit/ce82b740b8d0937a2faebe843d657dc08e76b441))
* **ui:** chat section ([#199](https://github.com/morphik-org/morphik-core/issues/199)) ([13d6d23](https://github.com/morphik-org/morphik-core/commit/13d6d23114bff8e14972eab25f516c33a150b2c6))
* **ui:** chat side bar  ([#195](https://github.com/morphik-org/morphik-core/issues/195)) ([a892208](https://github.com/morphik-org/morphik-core/commit/a8922087e9660b298c69b0f310e9cf8a83b0dc3e))
* **ui:** toast ([#193](https://github.com/morphik-org/morphik-core/issues/193)) ([bf605a9](https://github.com/morphik-org/morphik-core/commit/bf605a93b68a6a4254e6e8382b2d92a78b07a431))
* update prompt templates for clarity and version bump to 0.2.7 ([cf069d2](https://github.com/morphik-org/morphik-core/commit/cf069d265f4069ca539bee4824cca09c719e38a5))


### Chores

* bump @morphik/ui version to 0.3.27 ([8fd50a1](https://github.com/morphik-org/morphik-core/commit/8fd50a199b3413cce4e8e5a8025d45753949b25d))
* bump @morphik/ui version to 0.4.0 ([4722605](https://github.com/morphik-org/morphik-core/commit/47226058bf6efa2f61932e5c6fa29798d8980f99))
* sync repo ([d47a7ed](https://github.com/morphik-org/morphik-core/commit/d47a7ed2161f29a89342378a875fdae1fcccf25a))
* update SDK settings ([2087914](https://github.com/morphik-org/morphik-core/commit/208791492f43204d1917562ba047e7ec43225a30))


### Refactors

* replace hardcoded bucket path with direct string in storage key construction ([02af194](https://github.com/morphik-org/morphik-core/commit/02af194af299c348072cacff98dd60e740b703bc))
* update folder access checks to use _check_folder_model_access for consistency ([6f61f77](https://github.com/morphik-org/morphik-core/commit/6f61f77106bc4a9e7314294964c86d75602cbf63))
