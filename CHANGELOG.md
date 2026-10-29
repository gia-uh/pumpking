# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0]

First stable release.

### Added

- Basic strategies: `RegexChunking`, `FixedSizeChunking`, `ParagraphChunking`, `SentenceChunking`, `SlidingWindowChunking`, `AdaptiveChunking`.
- Advanced strategies: `HierarchicalChunking`, `EntityBasedChunking`, `SummaryChunking`, `TopicBasedChunking`, `ContextualChunking`, `ZettelkastenChunking`.
- Provider protocols in `pumpking.protocols`, and LLM implementations of each on a shared `LLMBackend` for OpenAI-compatible APIs.
- `PumpkingPipeline` with sequential steps (`>>`), branches (lists of steps) and annotations (`|` with `annotate`).
- `LLMBackend` parameters `timeout` and `max_retries`.
- `pumpking.exceptions.LLMBackendError`, raised when no API key is set or the endpoint cannot be reached. Providers no longer swallow these errors.
- `pumpking.__version__`.
- `examples/` with three scripts that run offline, and a fake OpenAI-compatible server for tests and demos.
- Continuous integration (pytest, ruff and the examples on Python 3.10 and 3.12) and a release workflow that publishes to PyPI with trusted publishing.

### Changed

- `LLMBackend(base_url=...)` is optional. It defaults to the `OPENAI_BASE_URL` environment variable, then to `https://api.openai.com/v1`.
- `SummaryChunking` sends all the blocks of a source to the provider in one call, as `SummaryProviderProtocol` defines, so `LLMSummaryProvider` batches them.
- The README Quick Start runs as written.
- Development dependencies moved from `tool.uv.dev-dependencies` to `dependency-groups`.

### Fixed

- `SummaryChunking` with `LLMSummaryProvider` failed with `AttributeError`.
- Strategies that accept a list of chunks failed with `AttributeError` as the first step of a pipeline.

[1.0.0]: https://github.com/gia-uh/pumpking/releases/tag/v1.0.0
