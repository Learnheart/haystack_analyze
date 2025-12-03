# Kiến Trúc Hệ Thống Haystack

## 1. Tổng Quan Kiến Trúc

Haystack sử dụng kiến trúc **Component-Based Architecture** với **Graph-Based Pipeline Execution**. Đây là kiến trúc modular cho phép linh hoạt và mở rộng dễ dàng.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            HAYSTACK FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        APPLICATION LAYER                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │  RAG Apps    │  │   QA Apps    │  │   Semantic Search    │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                        PIPELINE LAYER                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │                   Pipeline Orchestrator                     │ │   │
│  │  │  • Graph-based execution    • Async/Sync modes             │ │   │
│  │  │  • Component routing        • Error handling               │ │   │
│  │  │  • Breakpoints/Debugging    • Snapshot mechanism           │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                        COMPONENT LAYER                           │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │   │
│  │  │ Converters│ │ Embedders │ │ Retrievers│ │  Generators   │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │   │
│  │  │  Rankers  │ │  Builders │ │  Joiners  │ │   Routers     │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │   │
│  │  │ Extractors│ │ Evaluators│ │  Agents   │ │    Writers    │   │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                         DATA LAYER                               │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │   │
│  │  │   Document     │  │  ChatMessage   │  │     Answer       │  │   │
│  │  │   ByteStream   │  │    ToolCall    │  │     State        │  │   │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                       STORAGE LAYER                              │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │              Document Stores (Abstractions)                 │ │   │
│  │  │  • InMemoryDocumentStore (built-in)                        │ │   │
│  │  │  • External: Elasticsearch, OpenSearch, Weaviate, etc.     │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                      PROVIDER LAYER                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │   │
│  │  │  OpenAI  │ │  Azure   │ │  HF API  │ │  HF Transformers   │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │   │
│  │  │  Cohere  │ │ Bedrock  │ │SageMaker │ │    Local Models    │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Component System

### 2.1 Component Contract Pattern

Mỗi component trong Haystack tuân theo một contract chuẩn:

```python
from haystack import component

@component
class MyComponent:
    """
    Component Contract:
    1. Được đánh dấu bằng @component decorator
    2. Có __init__() để khởi tạo configuration
    3. Có run() method để xử lý data
    4. Optional: warm_up() để load resources nặng
    """

    def __init__(self, param: str):
        """Khởi tạo configuration - chạy khi tạo component"""
        self.param = param

    def warm_up(self):
        """
        Load resources nặng (models, connections).
        Chạy một lần trước khi pipeline run.
        """
        self.model = load_model()

    @component.output_types(output=str)
    def run(self, input_data: str) -> dict:
        """
        Xử lý data và trả về kết quả.
        Input: Các parameters được định nghĩa rõ ràng
        Output: Dictionary với keys tương ứng output_types
        """
        result = self.model.process(input_data)
        return {"output": result}
```

### 2.2 Input/Output Sockets

```
┌─────────────────────────────────────────────────┐
│                    COMPONENT                     │
│                                                  │
│  ┌──────────────┐         ┌──────────────────┐ │
│  │   INPUT      │         │      OUTPUT      │ │
│  │   SOCKETS    │   ───>  │      SOCKETS     │ │
│  │              │  run()  │                  │ │
│  │  • query     │         │  • documents     │ │
│  │  • filters   │         │  • score         │ │
│  │  • top_k     │         │                  │ │
│  └──────────────┘         └──────────────────┘ │
│                                                  │
└─────────────────────────────────────────────────┘

Sockets cho phép:
- Type checking tại connection time
- Validation của data flow
- Documentation tự động
```

## 3. Pipeline Architecture

### 3.1 Graph-Based Execution

```
                    ┌────────────────────────────────────────┐
                    │              PIPELINE                   │
                    │    (Directed Acyclic Graph - DAG)       │
                    │                                         │
Query ─────────────>│  ┌───────────┐     ┌───────────────┐   │
                    │  │ Embedder  │────>│   Retriever   │   │
                    │  └───────────┘     └───────┬───────┘   │
                    │                            │           │
                    │                    ┌───────▼───────┐   │
                    │                    │    Ranker     │   │
                    │                    └───────┬───────┘   │
                    │                            │           │
                    │  ┌───────────┐     ┌───────▼───────┐   │
                    │  │  Prompt   │<────│   Context     │   │
                    │  │  Builder  │     │   Builder     │   │
                    │  └─────┬─────┘     └───────────────┘   │
                    │        │                               │
                    │  ┌─────▼─────┐                         │
                    │  │ Generator │─────────────────────────│───> Answer
                    │  └───────────┘                         │
                    └────────────────────────────────────────┘
```

### 3.2 Pipeline Execution Flow

```
1. ADD COMPONENTS
   pipeline.add_component("embedder", TextEmbedder())
   pipeline.add_component("retriever", EmbeddingRetriever())
   pipeline.add_component("generator", OpenAIGenerator())

2. CONNECT COMPONENTS
   pipeline.connect("embedder.embedding", "retriever.query_embedding")
   pipeline.connect("retriever.documents", "generator.context")

3. VALIDATE PIPELINE
   - Check for cycles (must be DAG)
   - Validate socket types
   - Check required inputs

4. WARM UP (optional)
   pipeline.warm_up()
   - Loads models
   - Initializes connections

5. RUN PIPELINE
   result = pipeline.run({
       "embedder": {"text": "What is...?"},
       "generator": {"prompt": "..."}
   })
```

### 3.3 Pipeline Classes

```
┌─────────────────────────────────────────────────────────────┐
│                      PipelineBase                            │
│  (haystack/core/pipeline/base.py - 1,477 lines)             │
│                                                              │
│  • Graph management (add_component, connect)                 │
│  • Component registry                                        │
│  • Input/Output routing                                      │
│  • Serialization (to_dict, from_dict)                       │
│  • Breakpoint support                                        │
│  • Snapshot mechanism                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
┌──────────▼──────────┐        ┌──────────▼──────────┐
│      Pipeline       │        │   AsyncPipeline     │
│   (Synchronous)     │        │   (Asynchronous)    │
│                     │        │                     │
│  • Sequential exec  │        │  • Parallel exec    │
│  • Blocking calls   │        │  • Non-blocking     │
│  • Simple use cases │        │  • High throughput  │
└─────────────────────┘        └─────────────────────┘
```

## 4. Data Flow Architecture

### 4.1 Document-Centric Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Raw Files  │───>│  Converters  │───>│    Documents     │
│ PDF,DOCX... │    │              │    │  (Core Entity)   │
└─────────────┘    └──────────────┘    └────────┬─────────┘
                                                 │
                   ┌─────────────────────────────┤
                   │                             │
           ┌───────▼───────┐            ┌────────▼────────┐
           │  Preprocessors │            │    Embedders    │
           │  (split,clean) │            │  (vectorize)    │
           └───────┬───────┘            └────────┬────────┘
                   │                             │
           ┌───────▼────────────────────────────▼────────┐
           │              Document Store                  │
           │  • Store documents with metadata             │
           │  • Store embeddings (vectors)                │
           │  • Support filtering and search              │
           └──────────────────────┬──────────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
           ┌───────▼───────┐            ┌────────▼────────┐
           │   Retrievers   │            │    Rankers      │
           │  (BM25/Vector) │            │  (rerank docs)  │
           └───────┬───────┘            └────────┬────────┘
                   │                             │
           ┌───────▼─────────────────────────────▼───────┐
           │                  Generators                  │
           │  • Receive context documents                 │
           │  • Generate answers using LLM                │
           └──────────────────────┬──────────────────────┘
                                  │
                          ┌───────▼───────┐
                          │    Answer     │
                          │   (Output)    │
                          └───────────────┘
```

### 4.2 Document Structure

```python
@dataclass
class Document:
    id: str                           # Unique identifier
    content: Optional[str]            # Text content
    blob: Optional[ByteStream]        # Binary content
    meta: dict                        # Metadata
    score: Optional[float]            # Relevance score
    embedding: Optional[List[float]]  # Dense vector
    sparse_embedding: Optional[SparseEmbedding]  # Sparse vector

# Ví dụ
doc = Document(
    id="doc_001",
    content="Haystack is an LLM framework...",
    meta={
        "source": "website",
        "url": "https://haystack.deepset.ai",
        "date": "2024-01-15"
    },
    embedding=[0.1, 0.2, 0.3, ...]  # 768-dim vector
)
```

## 5. Serialization Architecture

### 5.1 Component Serialization

```
┌───────────────────┐     to_dict()      ┌───────────────────┐
│                   │  ───────────────>  │                   │
│    Component      │                    │   Dictionary      │
│    (Python)       │  <───────────────  │   (JSON-like)     │
│                   │    from_dict()     │                   │
└───────────────────┘                    └───────────────────┘
                                                  │
                                                  │ YAML/JSON
                                                  │
                                         ┌────────▼────────┐
                                         │                 │
                                         │   Persistent    │
                                         │   Storage       │
                                         │                 │
                                         └─────────────────┘

# Ví dụ serialization
component_dict = {
    "type": "haystack.components.generators.openai.OpenAIGenerator",
    "init_parameters": {
        "model": "gpt-4",
        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"]}
    }
}
```

### 5.2 Pipeline Serialization

```yaml
# pipeline.yaml
components:
  text_embedder:
    type: haystack.components.embedders.OpenAITextEmbedder
    init_parameters:
      model: text-embedding-3-small

  retriever:
    type: haystack.components.retrievers.InMemoryEmbeddingRetriever
    init_parameters:
      document_store:
        type: haystack.document_stores.InMemoryDocumentStore

  generator:
    type: haystack.components.generators.OpenAIGenerator
    init_parameters:
      model: gpt-4

connections:
  - sender: text_embedder.embedding
    receiver: retriever.query_embedding
  - sender: retriever.documents
    receiver: generator.context
```

## 6. Agent Architecture

### 6.1 Agent State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      STATE                                │   │
│  │  • messages: List[ChatMessage]                           │   │
│  │  • context: Dict[str, Any]                               │   │
│  │  • iteration: int                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   AGENT LOOP                              │   │
│  │                                                           │   │
│  │   ┌─────────┐    ┌──────────────┐    ┌─────────────┐    │   │
│  │   │ Receive │───>│   LLM Call   │───>│  Parse      │    │   │
│  │   │ Input   │    │  (Generator) │    │  Response   │    │   │
│  │   └─────────┘    └──────────────┘    └──────┬──────┘    │   │
│  │                                             │            │   │
│  │                    ┌────────────────────────┤            │   │
│  │                    │                        │            │   │
│  │              ┌─────▼─────┐          ┌───────▼───────┐   │   │
│  │              │  Text     │          │  Tool Call    │   │   │
│  │              │  Response │          │  Request      │   │   │
│  │              └─────┬─────┘          └───────┬───────┘   │   │
│  │                    │                        │            │   │
│  │                    │                ┌───────▼───────┐   │   │
│  │                    │                │ Execute Tools │   │   │
│  │                    │                └───────┬───────┘   │   │
│  │                    │                        │            │   │
│  │              ┌─────▼────────────────────────▼─────┐     │   │
│  │              │         Update State               │     │   │
│  │              └─────────────────┬──────────────────┘     │   │
│  │                                │                         │   │
│  │                        ┌───────▼───────┐                │   │
│  │                        │  Continue?    │                │   │
│  │                        │  (max_steps)  │                │   │
│  │                        └───────┬───────┘                │   │
│  │                                │                         │   │
│  │                    ┌───────────┴───────────┐            │   │
│  │                    │                       │            │   │
│  │              ┌─────▼─────┐         ┌───────▼───────┐   │   │
│  │              │   DONE    │         │   CONTINUE    │───┘   │
│  │              │  (Return) │         │   (Loop)      │       │
│  │              └───────────┘         └───────────────┘       │
│  │                                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Tool Calling

```
┌───────────────────────────────────────────────────────────────┐
│                        TOOL SYSTEM                             │
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │   Tool      │    │ Component   │    │   Pipeline      │   │
│  │ (Function)  │    │   Tool      │    │     Tool        │   │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘   │
│         │                  │                     │            │
│         └──────────────────┼─────────────────────┘            │
│                            │                                   │
│                    ┌───────▼───────┐                          │
│                    │    Toolset    │                          │
│                    │  (Collection) │                          │
│                    └───────┬───────┘                          │
│                            │                                   │
│                    ┌───────▼───────┐                          │
│                    │     Agent     │                          │
│                    │  (Executor)   │                          │
│                    └───────────────┘                          │
└───────────────────────────────────────────────────────────────┘

# Ví dụ Tool
@tool
def search_database(query: str, limit: int = 10) -> List[Document]:
    """Search the database for relevant documents."""
    return db.search(query, limit=limit)
```

## 7. Observability Architecture

### 7.1 Tracing

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRACING SYSTEM                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Pipeline Span                            │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │ │
│  │  │ Component A  │ │ Component B  │ │    Component C    │   │ │
│  │  │    Span      │ │    Span      │ │       Span        │   │ │
│  │  │              │ │              │ │                   │   │ │
│  │  │ • duration   │ │ • duration   │ │ • duration        │   │ │
│  │  │ • inputs     │ │ • inputs     │ │ • inputs          │   │ │
│  │  │ • outputs    │ │ • outputs    │ │ • outputs         │   │ │
│  │  │ • errors     │ │ • errors     │ │ • errors          │   │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│      ┌───────▼───────┐ ┌────▼────┐ ┌────────▼────────┐        │
│      │ OpenTelemetry │ │ DataDog │ │  Custom Tracer  │        │
│      └───────────────┘ └─────────┘ └─────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 8. Design Patterns Used

### 8.1 Decorator Pattern
- `@component` decorator để đánh dấu classes là pipeline components

### 8.2 Builder Pattern
- PromptBuilder, ChatPromptBuilder để xây dựng prompts
- Pipeline assembly với add_component() và connect()

### 8.3 Strategy Pattern
- Multiple implementations cho same interface (OpenAI vs HuggingFace Generators)
- Pluggable document stores

### 8.4 Template Method Pattern
- Component lifecycle: __init__ → warm_up → run

### 8.5 Observer Pattern
- Tracing và telemetry observers

### 8.6 Factory Pattern
- from_dict() methods để tạo components từ configuration
