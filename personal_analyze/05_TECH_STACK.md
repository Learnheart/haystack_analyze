# Tech Stack Haystack

## 1. Ngôn Ngữ & Runtime

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | >= 3.9 (hỗ trợ 3.9-3.13) | Ngôn ngữ chính |
| **Type Hints** | PEP 484, 585 | Static type checking |
| **Async/Await** | Python 3.9+ | Asynchronous execution |

## 2. Core Dependencies

### 2.1 Data Validation & Serialization

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `pydantic` | Latest | Data validation, settings management |
| `pyyaml` | Latest | YAML serialization cho pipelines |
| `jsonschema` | Latest | JSON schema validation |

### 2.2 Template Engine

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `Jinja2` | Latest | Template engine cho PromptBuilder |

### 2.3 Graph Processing

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `networkx` | Latest | Pipeline graph management, DAG operations |

### 2.4 Utilities

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `tqdm` | Latest | Progress bars |
| `tenacity` | != 8.4.0 | Retry logic với exponential backoff |
| `more-itertools` | Latest | Extended itertools (cho DocumentSplitter) |
| `lazy-imports` | Latest | Lazy module loading |
| `requests` | Latest | HTTP client |
| `numpy` | Latest (1.x & 2.x compatible) | Numerical operations |
| `python-dateutil` | Latest | Date/time parsing |
| `filetype` | Latest | MIME type detection |
| `docstring-parser` | Latest | Docstring parsing cho ComponentTool |

### 2.5 Telemetry

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `posthog` | != 3.12.0 | Anonymous usage statistics |

## 3. LLM Providers

### 3.1 OpenAI

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `openai` | >= 1.99.2 | OpenAI API client (GPT-4, GPT-3.5, DALL-E, Whisper) |

### 3.2 Hugging Face

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `transformers` | >= 4.55.4, < 4.57 | Local model inference |
| `huggingface_hub` | >= 0.27.0 | HF Inference API |
| `sentence-transformers` | >= 5.0.0 | Embedding models |
| `torch` | Latest | PyTorch backend |
| `sentencepiece` | Latest | Tokenization |

### 3.3 Azure

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `azure-identity` | Latest | Azure authentication |
| `azure-ai-formrecognizer` | >= 3.2.0b2 | Azure OCR/Document Intelligence |

## 4. Document Processing

### 4.1 PDF Processing

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `pypdf` | Latest | PDF text extraction |
| `pdfminer.six` | Latest | Advanced PDF parsing |
| `pypdfium2` | Latest | PDF to image conversion |

### 4.2 Office Documents

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `python-docx` | Latest | Word document processing |
| `python-pptx` | Latest | PowerPoint processing |
| `openpyxl` | Latest | Excel file reading |
| `python-oxmsg` | Latest | Outlook MSG files |
| `tabulate` | Latest | Table formatting |

### 4.3 Web Content

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `trafilatura` | Latest | HTML content extraction |
| `markdown-it-py` | Latest | Markdown parsing |
| `mdit_plain` | Latest | Markdown to plain text |

### 4.4 Other Formats

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `tika` | Latest | Universal file conversion (Apache Tika) |
| `jq` | Latest | JSON processing |
| `pandas` | Latest | CSV/Excel data processing |

## 5. NLP & Text Processing

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `nltk` | >= 3.9.1 | Text splitting, tokenization |
| `tiktoken` | Latest | OpenAI tokenizer |
| `langdetect` | Latest | Language detection |
| `spacy` | >= 3.8, < 3.9 | NER, NLP processing |

## 6. Audio Processing

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `openai-whisper` | >= 20231106 | Local Whisper transcription |

## 7. Image Processing

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `pillow` | Latest | Image manipulation |

## 8. Tracing & Observability

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `opentelemetry-sdk` | Latest | OpenTelemetry tracing |
| `ddtrace` | Latest | DataDog APM |
| `structlog` | Latest | Structured logging |

## 9. OpenAPI Integration

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `jsonref` | Latest | JSON reference resolution |
| `openapi3` | Latest | OpenAPI 3.0 parsing |
| `openapi-llm` | >= 0.4.1 | LLM-optimized OpenAPI |

## 10. HTTP & Networking

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `requests` | Latest | Synchronous HTTP |
| `httpx` | Latest (with http2) | Async HTTP, HTTP/2 support |

## 11. Date/Time

| Package | Phiên bản | Chức năng |
|---------|-----------|-----------|
| `arrow` | >= 1.3.0 | Date/time for Jinja2 extensions |

## 12. Development Tools

### 12.1 Build System

| Tool | Phiên bản | Chức năng |
|------|-----------|-----------|
| `hatchling` | >= 1.8.0 | Build backend |
| `hatch` | Latest | Project management |
| `uv` | Latest | Package installer |

### 12.2 Code Quality

| Tool | Phiên bản | Chức năng |
|------|-----------|-----------|
| `ruff` | Latest | Linting & formatting |
| `mypy` | Latest | Static type checking |
| `pylint` | Latest | Additional linting |
| `pre-commit` | Latest | Git hooks |

### 12.3 Testing

| Tool | Phiên bản | Chức năng |
|------|-----------|-----------|
| `pytest` | >= 6.0 | Test framework |
| `pytest-cov` | Latest | Coverage reporting |
| `pytest-asyncio` | Latest | Async test support |
| `pytest-bdd` | Latest | BDD testing |
| `pytest-rerunfailures` | Latest | Retry failed tests |
| `coverage` | Latest | Code coverage |

### 12.4 Documentation

| Tool | Phiên bản | Chức năng |
|------|-----------|-----------|
| `Docusaurus` | Latest | Documentation website |
| `reno` | Latest | Release notes |
| `haystack-pydoc-tools` | Latest | API documentation |

## 13. Infrastructure

### 13.1 CI/CD

- **GitHub Actions**: Automated testing, deployment
- **Coveralls**: Coverage reporting

### 13.2 Package Distribution

- **PyPI**: `haystack-ai` package
- **Conda-Forge**: `haystack-ai` package

## 14. Architecture Diagram - Tech Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│                     (RAG, QA, Search Applications)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                           HAYSTACK FRAMEWORK                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Core: Pipeline (networkx), Components (@component), Serialization  │   │
│  │ Data: pydantic, pyyaml, jsonschema                                  │   │
│  │ Templates: Jinja2                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼───────┐          ┌────────▼────────┐         ┌────────▼────────┐
│  LLM LAYER    │          │  EMBEDDING      │         │   DOCUMENT      │
│               │          │     LAYER       │         │   PROCESSING    │
│ • openai      │          │                 │         │                 │
│ • transformers│          │ • sentence-     │         │ • pypdf         │
│ • huggingface │          │   transformers  │         │ • python-docx   │
│   _hub        │          │ • openai        │         │ • trafilatura   │
│               │          │ • huggingface   │         │ • markdown-it   │
└───────────────┘          └─────────────────┘         └─────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                            NLP PROCESSING                                    │
│         nltk, tiktoken, langdetect, spacy, openai-whisper                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                             STORAGE LAYER                                    │
│   InMemory (built-in) │ Elasticsearch │ Weaviate │ Pinecone │ Qdrant │ ...  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                            OBSERVABILITY                                     │
│              opentelemetry-sdk, ddtrace, structlog, posthog                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────────┐
│                              INFRASTRUCTURE                                  │
│  Python 3.9+ │ hatchling (build) │ pytest (test) │ ruff/mypy (quality)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 15. Version Requirements Summary

```toml
# Minimum versions
python = ">=3.9"
openai = ">=1.99.2"
transformers = ">=4.55.4,<4.57"
huggingface_hub = ">=0.27.0"
sentence-transformers = ">=5.0.0"
azure-ai-formrecognizer = ">=3.2.0b2"
nltk = ">=3.9.1"
arrow = ">=1.3.0"
openapi-llm = ">=0.4.1"
openai-whisper = ">=20231106"
typing_extensions = ">=4.7"
spacy = ">=3.8,<3.9"

# Excluded versions (bugs)
tenacity != "8.4.0"
posthog != "3.12.0"
```

## 16. Optional Dependencies by Feature

### 16.1 Basic RAG
```
haystack-ai
openai
sentence-transformers
```

### 16.2 Document Processing
```
pypdf OR pdfminer.six
python-docx
trafilatura
```

### 16.3 Audio Transcription
```
openai-whisper  # Local
# OR use OpenAI API
```

### 16.4 Advanced NLP
```
spacy
nltk
langdetect
```

### 16.5 Tracing
```
opentelemetry-sdk  # OR
ddtrace
```

### 16.6 Azure Integration
```
azure-identity
azure-ai-formrecognizer
```
