# Cấu Trúc Thư Mục Haystack

## 1. Tổng Quan Cấu Trúc

```
haystack_analyze/
├── haystack/                    # Source code chính
│   ├── __init__.py             # Public API exports
│   ├── core/                   # Core framework components
│   ├── components/             # 20+ loại components
│   ├── dataclasses/            # Data structures
│   ├── document_stores/        # Document storage
│   ├── evaluation/             # Evaluation tools
│   ├── marshal/                # Serialization
│   ├── telemetry/              # Usage tracking
│   ├── tracing/                # Observability
│   ├── testing/                # Test utilities
│   ├── tools/                  # Tool/function utilities
│   ├── utils/                  # Helper functions
│   └── data/                   # Static data files
├── test/                       # Unit & Integration tests
├── e2e/                        # End-to-end tests
├── docs-website/               # Documentation (Docusaurus)
├── examples/                   # Example references
├── docker/                     # Docker configurations
├── pydoc/                      # API documentation generation
├── releasenotes/               # Release notes
├── proposals/                  # Design proposals
├── images/                     # Documentation images
├── pyproject.toml              # Project configuration
├── VERSION.txt                 # Version number
└── README.md                   # Project readme
```

## 2. Chi Tiết Thư Mục Haystack

### 2.1 `/haystack/core/` - Core Framework

```
core/
├── __init__.py
├── errors.py                   # Custom exceptions
├── serialization.py            # Serialization utilities
├── component/                  # Component system
│   ├── __init__.py
│   ├── component.py            # @component decorator (630 lines)
│   ├── connection.py           # Component connections
│   ├── sockets.py              # Input/Output sockets
│   └── types.py                # Type annotations
├── pipeline/                   # Pipeline orchestration
│   ├── __init__.py
│   ├── base.py                 # PipelineBase class (1,477 lines)
│   ├── pipeline.py             # Synchronous Pipeline
│   ├── async_pipeline.py       # Asynchronous Pipeline
│   ├── template.py             # Pipeline templates
│   ├── draw/                   # Pipeline visualization
│   └── features/               # Advanced features (loops, breakpoints)
└── super_component/            # Complex reusable patterns
    ├── __init__.py
    └── super_component.py      # SuperComponent class
```

### 2.2 `/haystack/components/` - Components (~20 categories)

```
components/
├── __init__.py
├── agents/                     # Agentic workflows
│   ├── agent.py               # Agent class
│   └── state/                 # State management
├── audio/                      # Audio processing
│   ├── whisper_local.py       # Local Whisper
│   └── whisper_remote.py      # Remote Whisper API
├── builders/                   # Prompt/Answer building
│   ├── prompt_builder.py      # Jinja2 template builder
│   ├── chat_prompt_builder.py # Chat prompt builder
│   └── answer_builder.py      # Answer aggregation
├── caching/                    # Caching components
│   └── cache_checker.py
├── classifiers/                # Document classification
│   ├── document_language_classifier.py
│   └── zero_shot_document_classifier.py
├── connectors/                 # External API connectors
│   ├── openapi.py
│   └── openapi_service.py
├── converters/                 # File format converters
│   ├── pypdf.py               # PDF (PyPDF)
│   ├── pdfminer.py            # PDF (PDFMiner)
│   ├── docx.py                # Word documents
│   ├── html.py                # HTML
│   ├── markdown.py            # Markdown
│   ├── pptx.py                # PowerPoint
│   ├── xlsx.py                # Excel
│   ├── csv.py                 # CSV
│   ├── json.py                # JSON
│   ├── txt.py                 # Plain text
│   ├── azure.py               # Azure OCR
│   ├── tika.py                # Apache Tika
│   ├── msg.py                 # Outlook MSG
│   ├── image/                 # Image converters
│   └── multi_file_converter.py
├── embedders/                  # Embedding generation
│   ├── openai_text_embedder.py
│   ├── openai_document_embedder.py
│   ├── azure_text_embedder.py
│   ├── azure_document_embedder.py
│   ├── sentence_transformers_*.py
│   ├── hugging_face_api_*.py
│   ├── backends/              # Embedding backends
│   ├── image/                 # Image embedders
│   └── types/                 # Type protocols
├── evaluators/                 # Evaluation metrics
│   ├── answer_exact_match.py
│   ├── context_relevance.py
│   ├── document_map.py
│   ├── document_mrr.py
│   ├── document_recall.py
│   ├── document_ndcg.py
│   ├── faithfulness.py
│   ├── llm_evaluator.py
│   └── sas_evaluator.py
├── extractors/                 # Information extraction
│   ├── llm_metadata_extractor.py
│   ├── named_entity_extractor.py
│   ├── regex_text_extractor.py
│   └── image/
├── fetchers/                   # Content fetching
│   └── link_content.py
├── generators/                 # Text generation
│   ├── openai.py              # OpenAI Generator
│   ├── azure.py               # Azure OpenAI Generator
│   ├── hugging_face_api.py    # HF API Generator
│   ├── hugging_face_local.py  # HF Local Generator
│   ├── openai_dalle.py        # DALL-E Image Generator
│   └── chat/                  # Chat Generators
│       ├── openai.py
│       ├── azure.py
│       ├── hugging_face_*.py
│       └── fallback.py
├── joiners/                    # Result joining
│   ├── answer_joiner.py
│   ├── document_joiner.py
│   ├── list_joiner.py
│   └── branch_joiner.py
├── preprocessors/              # Document preprocessing
│   ├── document_splitter.py
│   ├── recursive_splitter.py
│   ├── hierarchical_splitter.py
│   ├── document_cleaner.py
│   ├── text_cleaner.py
│   ├── nltk_document_splitter.py
│   └── csv_document_*/
├── rankers/                    # Document ranking
│   ├── meta_field_ranker.py
│   ├── lost_in_the_middle_ranker.py
│   └── huggingface_tei_ranker.py
├── retrievers/                 # Document retrieval
│   ├── in_memory/
│   │   ├── bm25_retriever.py
│   │   └── embedding_retriever.py
│   ├── filter_retriever.py
│   ├── sentence_window_retriever.py
│   └── auto_merging_retriever.py
├── routers/                    # Conditional routing
│   ├── file_type_router.py
│   ├── metadata_router.py
│   ├── text_language_router.py
│   └── transformers_text_router.py
├── samplers/                   # Sampling
│   └── top_p_sampler.py
├── validators/                 # Validation
│   └── json_schema_validator.py
├── websearch/                  # Web search
│   └── searchapi.py
└── writers/                    # Document writing
    └── document_writer.py
```

### 2.3 `/haystack/dataclasses/` - Data Structures

```
dataclasses/
├── __init__.py
├── answer.py                   # Answer, ExtractedAnswer, GeneratedAnswer
├── byte_stream.py              # ByteStream (binary data)
├── chat_message.py             # ChatMessage, ChatRole
├── document.py                 # Document (core data unit)
├── image_content.py            # ImageContent
├── sparse_embedding.py         # SparseEmbedding
├── state.py                    # State (agent workflows)
├── streaming_chunk.py          # StreamingChunk
└── tool_call.py                # ToolCall, ToolCallResult
```

### 2.4 `/haystack/document_stores/` - Document Storage

```
document_stores/
├── __init__.py
├── errors.py                   # Document store exceptions
├── in_memory/                  # In-memory implementation
│   ├── __init__.py
│   └── document_store.py
└── types/                      # Abstract interfaces
    ├── __init__.py
    ├── filter_policy.py
    └── policy.py
```

### 2.5 Thư Mục Khác

```
/haystack/marshal/              # Serialization
├── protocol.py
└── yaml.py

/haystack/telemetry/            # Telemetry
└── telemetry.py                # PostHog integration

/haystack/tracing/              # Tracing
├── opentelemetry.py
├── datadog.py
└── tracer.py

/haystack/tools/                # Tool utilities
├── tool.py
├── component_tool.py
├── pipeline_tool.py
├── from_function.py
└── toolset.py

/haystack/utils/                # Utilities
├── auth.py
├── azure.py
├── device.py
├── filters.py
├── http_client.py
├── jinja2_extensions.py
├── type_serialization.py
└── type_utils.py
```

## 3. Thư Mục Test

```
test/
├── conftest.py                 # Pytest fixtures
├── core/                       # Core tests
│   ├── component/
│   └── pipeline/
├── components/                 # Component tests
│   ├── agents/
│   ├── builders/
│   ├── converters/
│   ├── embedders/
│   ├── generators/
│   └── ...
├── dataclasses/                # Data class tests
├── document_stores/            # Document store tests
└── utils/                      # Utility tests

e2e/                            # End-to-end tests
├── pipelines/
└── samples/
```

## 4. Documentation & Config

```
docs-website/                   # Docusaurus documentation site
├── docs/
├── api/
└── docusaurus.config.js

pydoc/                          # API documentation generation
├── README.md
└── config/

releasenotes/                   # Release notes (reno)
└── notes/

proposals/                      # Design proposals
└── text/
```

## 5. Metrics

| Thư mục | Số file Python | Mô tả |
|---------|----------------|-------|
| `haystack/` | ~265 files | Main source code |
| `test/` | ~236 files | Unit & integration tests |
| `e2e/` | ~15 files | End-to-end tests |
| **Total** | **~516 files** | - |
