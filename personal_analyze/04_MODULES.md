# Chi Tiết Các Module Haystack

## 1. Core Module (`/haystack/core/`)

### 1.1 Component System

**File chính:** `component/component.py` (630 lines)

```python
# Decorator @component
@component
class MyComponent:
    @component.output_types(documents=List[Document])
    def run(self, query: str) -> Dict[str, List[Document]]:
        return {"documents": [...]}
```

**Chức năng:**
- `@component` decorator: Đánh dấu class là pipeline component
- Input socket definition: Tự động từ run() parameters
- Output socket definition: Từ `@component.output_types()`
- Lifecycle hooks: `__init__`, `warm_up`, `run`
- Serialization: `to_dict()`, `from_dict()`

**Sockets (`sockets.py`):**
- `InputSocket`: Định nghĩa input port với type hints
- `OutputSocket`: Định nghĩa output port với type hints
- Type validation tại connection time

### 1.2 Pipeline System

**File chính:** `pipeline/base.py` (1,477 lines)

```python
class PipelineBase:
    """Base class for Pipeline orchestration"""

    def add_component(self, name: str, instance: Component):
        """Thêm component vào pipeline"""

    def connect(self, sender: str, receiver: str):
        """Kết nối output của component này với input của component khác"""

    def run(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Chạy pipeline với input data"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline thành dictionary"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineBase":
        """Deserialize pipeline từ dictionary"""
```

**Features:**
- Graph-based execution engine
- Component routing và validation
- Breakpoint support cho debugging
- Snapshot mechanism cho agent workflows
- Async support (`async_pipeline.py`)

### 1.3 Super Component

**File:** `super_component/super_component.py`

```python
@super_component
class RAGPipeline:
    """Complex reusable component pattern"""

    def __init__(self, document_store, model):
        self.pipeline = Pipeline()
        self.pipeline.add_component("retriever", ...)
        self.pipeline.add_component("generator", ...)
```

**Chức năng:**
- Wrap một pipeline thành một component
- Reusable complex patterns
- Agent pattern support

---

## 2. Components Module (`/haystack/components/`)

### 2.1 Generators (Text Generation)

**Location:** `/components/generators/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `OpenAIGenerator` | `openai.py` | Text generation với OpenAI API |
| `AzureOpenAIGenerator` | `azure.py` | Text generation với Azure OpenAI |
| `HuggingFaceLocalGenerator` | `hugging_face_local.py` | Local HuggingFace models |
| `HuggingFaceAPIGenerator` | `hugging_face_api.py` | HuggingFace Inference API |
| `DALLEImageGenerator` | `openai_dalle.py` | Image generation với DALL-E |

**Chat Generators:** `/components/generators/chat/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `OpenAIChatGenerator` | `openai.py` | Chat với OpenAI (gpt-4, gpt-3.5) |
| `AzureOpenAIChatGenerator` | `azure.py` | Chat với Azure OpenAI |
| `HuggingFaceLocalChatGenerator` | `hugging_face_local.py` | Local chat models |
| `FallbackGenerator` | `fallback.py` | Fallback khi primary fails |

```python
# Ví dụ sử dụng
from haystack.components.generators import OpenAIGenerator

generator = OpenAIGenerator(
    model="gpt-4",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    generation_kwargs={"temperature": 0.7}
)
result = generator.run(prompt="Explain quantum computing")
```

### 2.2 Embedders (Vector Embedding)

**Location:** `/components/embedders/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `OpenAITextEmbedder` | `openai_text_embedder.py` | Embed single text |
| `OpenAIDocumentEmbedder` | `openai_document_embedder.py` | Embed documents |
| `AzureOpenAITextEmbedder` | `azure_text_embedder.py` | Azure embeddings |
| `SentenceTransformersTextEmbedder` | `sentence_transformers_text_embedder.py` | Local ST models |
| `SentenceTransformersDocumentEmbedder` | `sentence_transformers_document_embedder.py` | Embed docs locally |
| `HuggingFaceAPITextEmbedder` | `hugging_face_api_text_embedder.py` | HF API embeddings |

**Sparse Embedders:**
- `SentenceTransformersSparseTextEmbedder`
- `SentenceTransformersSparseDocumentEmbedder`

```python
# Ví dụ
from haystack.components.embedders import SentenceTransformersTextEmbedder

embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
embedder.warm_up()  # Load model
result = embedder.run(text="Hello world")
# result["embedding"] = [0.1, 0.2, ...]  # 384-dim vector
```

### 2.3 Retrievers (Document Retrieval)

**Location:** `/components/retrievers/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `InMemoryBM25Retriever` | `in_memory/bm25_retriever.py` | Keyword-based (BM25) |
| `InMemoryEmbeddingRetriever` | `in_memory/embedding_retriever.py` | Vector similarity |
| `FilterRetriever` | `filter_retriever.py` | Metadata filtering |
| `SentenceWindowRetriever` | `sentence_window_retriever.py` | Contextual chunks |
| `AutoMergingRetriever` | `auto_merging_retriever.py` | Adaptive merging |

```python
# Ví dụ RAG retrieval
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore

doc_store = InMemoryDocumentStore()
retriever = InMemoryEmbeddingRetriever(
    document_store=doc_store,
    top_k=5
)
result = retriever.run(query_embedding=[0.1, 0.2, ...])
```

### 2.4 Converters (File Conversion)

**Location:** `/components/converters/`

| Component | File | Input | Output |
|-----------|------|-------|--------|
| `PyPDFToDocument` | `pypdf.py` | PDF files | Documents |
| `PDFMinerToDocument` | `pdfminer.py` | PDF files | Documents |
| `DOCXToDocument` | `docx.py` | Word docs | Documents |
| `HTMLToDocument` | `html.py` | HTML files | Documents |
| `MarkdownToDocument` | `markdown.py` | Markdown | Documents |
| `PPTXToDocument` | `pptx.py` | PowerPoint | Documents |
| `XLSXToDocument` | `xlsx.py` | Excel | Documents |
| `CSVToDocument` | `csv.py` | CSV files | Documents |
| `JSONConverter` | `json.py` | JSON files | Documents |
| `TikaDocumentConverter` | `tika.py` | Multiple | Documents |
| `AzureOCRDocumentConverter` | `azure.py` | Images/Scans | Documents |

```python
# Ví dụ convert PDF
from haystack.components.converters import PyPDFToDocument

converter = PyPDFToDocument()
result = converter.run(sources=["document.pdf"])
# result["documents"] = [Document(...), ...]
```

### 2.5 Preprocessors (Document Processing)

**Location:** `/components/preprocessors/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `DocumentSplitter` | `document_splitter.py` | Split by words/sentences/pages |
| `RecursiveDocumentSplitter` | `recursive_splitter.py` | Recursive splitting |
| `HierarchicalDocumentSplitter` | `hierarchical_splitter.py` | Hierarchical chunks |
| `DocumentCleaner` | `document_cleaner.py` | Clean text content |
| `TextCleaner` | `text_cleaner.py` | Clean raw text |
| `NLTKDocumentSplitter` | `nltk_document_splitter.py` | NLTK-based splitting |

```python
# Ví dụ splitting
from haystack.components.preprocessors import DocumentSplitter

splitter = DocumentSplitter(
    split_by="sentence",
    split_length=3,
    split_overlap=1
)
result = splitter.run(documents=[doc])
```

### 2.6 Builders (Prompt/Answer Building)

**Location:** `/components/builders/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `PromptBuilder` | `prompt_builder.py` | Jinja2 template builder |
| `ChatPromptBuilder` | `chat_prompt_builder.py` | Chat message builder |
| `AnswerBuilder` | `answer_builder.py` | Aggregate answers |

```python
# Ví dụ Prompt Builder
from haystack.components.builders import PromptBuilder

builder = PromptBuilder(template="""
Given these documents: {{ documents }}
Answer the question: {{ question }}
""")

result = builder.run(
    documents=docs,
    question="What is Haystack?"
)
```

### 2.7 Rankers (Document Ranking)

**Location:** `/components/rankers/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `MetaFieldRanker` | `meta_field_ranker.py` | Rank by metadata field |
| `LostInTheMiddleRanker` | `lost_in_the_middle_ranker.py` | Reorder for LLM attention |
| `HuggingFaceTEIRanker` | `huggingface_tei_ranker.py` | Semantic reranking |

### 2.8 Routers (Conditional Routing)

**Location:** `/components/routers/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `FileTypeRouter` | `file_type_router.py` | Route by file type |
| `MetadataRouter` | `metadata_router.py` | Route by metadata |
| `TextLanguageRouter` | `text_language_router.py` | Route by language |
| `TransformersTextRouter` | `transformers_text_router.py` | ML-based routing |

### 2.9 Joiners (Result Aggregation)

**Location:** `/components/joiners/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `DocumentJoiner` | `document_joiner.py` | Merge document lists |
| `AnswerJoiner` | `answer_joiner.py` | Merge answers |
| `ListJoiner` | `list_joiner.py` | Generic list joining |
| `BranchJoiner` | `branch_joiner.py` | Join pipeline branches |

### 2.10 Extractors (Information Extraction)

**Location:** `/components/extractors/`

| Component | File | Chức năng |
|-----------|------|-----------|
| `LLMMetadataExtractor` | `llm_metadata_extractor.py` | Extract metadata với LLM |
| `NamedEntityExtractor` | `named_entity_extractor.py` | NER extraction |
| `RegexTextExtractor` | `regex_text_extractor.py` | Pattern-based extraction |

### 2.11 Evaluators (Quality Assessment)

**Location:** `/components/evaluators/`

| Component | File | Metrics |
|-----------|------|---------|
| `AnswerExactMatch` | `answer_exact_match.py` | Exact match accuracy |
| `ContextRelevance` | `context_relevance.py` | Context relevance score |
| `DocumentMAP` | `document_map.py` | Mean Average Precision |
| `DocumentMRR` | `document_mrr.py` | Mean Reciprocal Rank |
| `DocumentRecall` | `document_recall.py` | Recall@K |
| `DocumentNDCG` | `document_ndcg.py` | NDCG score |
| `Faithfulness` | `faithfulness.py` | Answer faithfulness |
| `SASEvaluator` | `sas_evaluator.py` | Semantic Answer Similarity |

### 2.12 Agents (Agentic Workflows)

**Location:** `/components/agents/`

```python
from haystack.components.agents import Agent

agent = Agent(
    generator=OpenAIChatGenerator(),
    tools=[search_tool, calculator_tool],
    max_steps=10
)

result = agent.run(messages=[
    ChatMessage.from_user("What's the weather in Tokyo?")
])
```

**State Management:** `/components/agents/state/`
- `State` class: Quản lý agent state
- `state_utils`: Utilities cho state operations

### 2.13 Other Components

**Audio:** `/components/audio/`
- `LocalWhisperTranscriber`: Local Whisper model
- `RemoteWhisperTranscriber`: OpenAI Whisper API

**Classifiers:** `/components/classifiers/`
- `DocumentLanguageClassifier`: Detect document language
- `ZeroShotDocumentClassifier`: Zero-shot classification

**Connectors:** `/components/connectors/`
- `OpenAPIConnector`: Connect to OpenAPI services
- `OpenAPIServiceConnector`: Legacy connector

**Fetchers:** `/components/fetchers/`
- `LinkContentFetcher`: Fetch content from URLs

**Writers:** `/components/writers/`
- `DocumentWriter`: Write documents to store

**Validators:** `/components/validators/`
- `JsonSchemaValidator`: Validate JSON schema

---

## 3. DataClasses Module (`/haystack/dataclasses/`)

### 3.1 Document

**File:** `document.py`

```python
@dataclass
class Document:
    id: str = ""                                    # Unique ID
    content: Optional[str] = None                   # Text content
    blob: Optional[ByteStream] = None               # Binary data
    meta: dict = field(default_factory=dict)        # Metadata
    score: Optional[float] = None                   # Relevance score
    embedding: Optional[List[float]] = None         # Dense vector
    sparse_embedding: Optional[SparseEmbedding] = None  # Sparse vector
```

### 3.2 ChatMessage

**File:** `chat_message.py`

```python
@dataclass
class ChatMessage:
    role: ChatRole                          # user, assistant, system, tool
    content: Union[str, List[ContentPart]]  # Message content
    name: Optional[str] = None              # Tool name
    tool_calls: List[ToolCall] = []         # Tool call requests
    tool_call_results: List[ToolCallResult] = []  # Tool results
```

### 3.3 Answer

**File:** `answer.py`

```python
@dataclass
class Answer:
    data: Any                    # Answer data
    query: str                   # Original query
    meta: dict = {}              # Metadata

@dataclass
class GeneratedAnswer(Answer):
    documents: List[Document]    # Source documents

@dataclass
class ExtractedAnswer(Answer):
    document: Document           # Source document
    start: int                   # Start position
    end: int                     # End position
```

### 3.4 Other DataClasses

- `ByteStream`: Binary data wrapper
- `ImageContent`: Image với MIME type
- `SparseEmbedding`: Sparse vector representation
- `State`: Agent workflow state
- `StreamingChunk`: Streaming response data
- `ToolCall`, `ToolCallResult`: Tool calling data

---

## 4. Document Stores Module (`/haystack/document_stores/`)

### 4.1 In-Memory Store

**File:** `in_memory/document_store.py`

```python
class InMemoryDocumentStore:
    """Default document store for development/testing"""

    def write_documents(self, documents: List[Document]) -> int:
        """Store documents"""

    def filter_documents(self, filters: Dict) -> List[Document]:
        """Filter by metadata"""

    def bm25_retrieval(self, query: str, top_k: int) -> List[Document]:
        """Keyword search"""

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Document]:
        """Vector similarity search"""
```

### 4.2 External Stores (via integrations)

- Elasticsearch
- OpenSearch
- Weaviate
- Pinecone
- Qdrant
- Milvus
- ChromaDB
- pgvector
- MongoDB Atlas

---

## 5. Tools Module (`/haystack/tools/`)

### 5.1 Tool Definition

**File:** `tool.py`

```python
from haystack.tools import Tool

# Định nghĩa tool từ function
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return perform_search(query, max_results)

# Hoặc manually
tool = Tool(
    name="search_web",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    },
    function=search_web
)
```

### 5.2 ComponentTool

**File:** `component_tool.py`

```python
from haystack.tools import ComponentTool

# Convert component thành tool
tool = ComponentTool(
    component=WebSearchComponent(),
    name="web_search",
    description="Search the web"
)
```

### 5.3 Toolset

**File:** `toolset.py`

```python
from haystack.tools import Toolset

toolset = Toolset(tools=[
    search_tool,
    calculator_tool,
    weather_tool
])

# Sử dụng với Agent
agent = Agent(
    generator=generator,
    tools=toolset
)
```

---

## 6. Utils Module (`/haystack/utils/`)

| File | Chức năng |
|------|-----------|
| `auth.py` | Authentication utilities |
| `azure.py` | Azure-specific helpers |
| `device.py` | GPU/CPU device selection |
| `filters.py` | Advanced filter syntax |
| `http_client.py` | HTTP client utilities |
| `jinja2_extensions.py` | Custom Jinja2 extensions |
| `type_serialization.py` | Type serialization |
| `type_utils.py` | Type checking utilities |
| `url_validation.py` | URL validation |
| `callable_serialization.py` | Function serialization |

---

## 7. Telemetry & Tracing

### 7.1 Telemetry

**File:** `telemetry/telemetry.py`

- Anonymous usage statistics
- PostHog integration
- Opt-out available

### 7.2 Tracing

**Files:** `tracing/`

| File | Chức năng |
|------|-----------|
| `opentelemetry.py` | OpenTelemetry integration |
| `datadog.py` | DataDog APM integration |
| `tracer.py` | Base tracer interface |

```python
# Enable tracing
from haystack.tracing import enable_tracing

enable_tracing(
    OpenTelemetryTracer(
        service_name="my-rag-app"
    )
)
```

---

## 8. Testing Module (`/haystack/testing/`)

**Chức năng:**
- Sample components cho testing
- Mock implementations
- Test utilities

```python
from haystack.testing import DocumentBuilder

# Tạo test documents
docs = DocumentBuilder.build_documents(
    count=10,
    with_embeddings=True
)
```
