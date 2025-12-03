# Haystack Analysis Documentation

## Giới Thiệu

Thư mục này chứa tài liệu phân tích chi tiết về **Haystack** - một framework LLM end-to-end được phát triển bởi deepset.

## Danh Sách Tài Liệu

| File | Mô tả |
|------|-------|
| [01_OVERVIEW.md](./01_OVERVIEW.md) | Tổng quan dự án, mục đích sử dụng, triết lý thiết kế |
| [02_DIRECTORY_STRUCTURE.md](./02_DIRECTORY_STRUCTURE.md) | Cấu trúc cây thư mục chi tiết |
| [03_ARCHITECTURE.md](./03_ARCHITECTURE.md) | Kiến trúc hệ thống, design patterns |
| [04_MODULES.md](./04_MODULES.md) | Chi tiết các module và components |
| [05_TECH_STACK.md](./05_TECH_STACK.md) | Phân tích tech stack và dependencies |
| [06_SEQUENCE_DIAGRAMS.md](./06_SEQUENCE_DIAGRAMS.md) | Sequence diagrams (Mermaid) mô tả luồng hoạt động |

## Tóm Tắt

### Haystack là gì?

**Haystack** là framework cho phép xây dựng các ứng dụng AI sử dụng:
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Document Processing
- Agentic Workflows

### Kiến Trúc Chính

```
┌─────────────────────────────────────────────────┐
│              APPLICATION LAYER                   │
│         (RAG, QA, Search, Agents)               │
├─────────────────────────────────────────────────┤
│              PIPELINE LAYER                      │
│         (Orchestration, Routing)                │
├─────────────────────────────────────────────────┤
│              COMPONENT LAYER                     │
│  (Generators, Embedders, Retrievers, etc.)      │
├─────────────────────────────────────────────────┤
│              DATA LAYER                          │
│     (Document, ChatMessage, Answer)             │
├─────────────────────────────────────────────────┤
│              STORAGE LAYER                       │
│         (Document Stores)                       │
├─────────────────────────────────────────────────┤
│              PROVIDER LAYER                      │
│    (OpenAI, Azure, HuggingFace, etc.)          │
└─────────────────────────────────────────────────┘
```

### Tech Stack Chính

- **Language**: Python 3.9+
- **Core**: pydantic, Jinja2, networkx, pyyaml
- **LLM**: openai, transformers, sentence-transformers
- **Build**: hatchling, hatch
- **Quality**: ruff, mypy, pytest

### Components Chính (20+ loại)

1. **Generators** - Text/Chat generation (OpenAI, HuggingFace, Azure)
2. **Embedders** - Vector embeddings
3. **Retrievers** - Document retrieval (BM25, Vector)
4. **Converters** - File format conversion (PDF, DOCX, HTML, etc.)
5. **Preprocessors** - Document splitting, cleaning
6. **Rankers** - Document ranking
7. **Builders** - Prompt/Answer building
8. **Agents** - Agentic workflows with tool calling
9. **Evaluators** - Quality metrics
10. **Routers** - Conditional routing

### Sequence Diagrams

Tài liệu bao gồm các sequence diagrams mô tả:
- RAG Pipeline flow
- Document Ingestion
- Chat Pipeline
- Agent với Tool Calling
- Component Lifecycle
- Error Handling
- Async Execution

## Cách Xem Diagrams

Các sequence diagrams sử dụng **Mermaid** syntax. Để xem:

1. **GitHub**: Tự động render Mermaid trong markdown
2. **VS Code**: Cài extension "Mermaid Preview"
3. **Online**: Paste vào [Mermaid Live Editor](https://mermaid.live/)

## Thống Kê

| Metric | Giá trị |
|--------|---------|
| Python files (source) | ~265 |
| Python files (tests) | ~236 |
| Component types | 20+ |
| LLM Providers | 5+ |
| Document formats | 15+ |

## Tham Khảo

- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [Haystack GitHub](https://github.com/deepset-ai/haystack)
- [Haystack Cookbook](https://haystack.deepset.ai/cookbook)
- [deepset.ai](https://www.deepset.ai/)

---

*Tài liệu được tạo để phục vụ mục đích phân tích và học tập.*
