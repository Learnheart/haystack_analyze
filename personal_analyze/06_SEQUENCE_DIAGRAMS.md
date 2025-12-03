# Sequence Diagrams - Luồng Hoạt Động Haystack

## 1. RAG Pipeline - Retrieval Augmented Generation

### 1.1 Basic RAG Flow

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant TextEmbedder
    participant Retriever
    participant PromptBuilder
    participant Generator
    participant DocumentStore

    User->>Pipeline: run(query="What is Haystack?")

    Note over Pipeline: Phase 1: Embedding
    Pipeline->>TextEmbedder: run(text=query)
    TextEmbedder->>TextEmbedder: encode(query)
    TextEmbedder-->>Pipeline: {embedding: [0.1, 0.2, ...]}

    Note over Pipeline: Phase 2: Retrieval
    Pipeline->>Retriever: run(query_embedding)
    Retriever->>DocumentStore: embedding_retrieval(embedding, top_k=5)
    DocumentStore-->>Retriever: [doc1, doc2, doc3, doc4, doc5]
    Retriever-->>Pipeline: {documents: [...]}

    Note over Pipeline: Phase 3: Prompt Building
    Pipeline->>PromptBuilder: run(documents, query)
    PromptBuilder->>PromptBuilder: render_template()
    PromptBuilder-->>Pipeline: {prompt: "Given context..."}

    Note over Pipeline: Phase 4: Generation
    Pipeline->>Generator: run(prompt)
    Generator->>Generator: call_llm(prompt)
    Generator-->>Pipeline: {replies: ["Haystack is..."]}

    Pipeline-->>User: {answer: "Haystack is..."}
```

### 1.2 RAG với Ranker

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Embedder
    participant Retriever
    participant Ranker
    participant PromptBuilder
    participant Generator

    User->>Pipeline: run(query)

    Pipeline->>Embedder: run(text=query)
    Embedder-->>Pipeline: embedding

    Pipeline->>Retriever: run(query_embedding, top_k=20)
    Retriever-->>Pipeline: documents (20 docs)

    Note over Ranker: Rerank for relevance
    Pipeline->>Ranker: run(query, documents)
    Ranker->>Ranker: score_documents()
    Ranker-->>Pipeline: ranked_documents (top 5)

    Pipeline->>PromptBuilder: run(documents, query)
    PromptBuilder-->>Pipeline: prompt

    Pipeline->>Generator: run(prompt)
    Generator-->>Pipeline: answer

    Pipeline-->>User: answer
```

---

## 2. Document Ingestion Pipeline

### 2.1 Basic Document Ingestion

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Converter
    participant Splitter
    participant Embedder
    participant Writer
    participant DocumentStore

    User->>Pipeline: run(sources=["file.pdf"])

    Note over Pipeline: Phase 1: Conversion
    Pipeline->>Converter: run(sources)
    Converter->>Converter: extract_text(pdf)
    Converter-->>Pipeline: {documents: [Document(content="...")]}

    Note over Pipeline: Phase 2: Splitting
    Pipeline->>Splitter: run(documents)
    Splitter->>Splitter: split_by_sentence()
    Splitter-->>Pipeline: {documents: [chunk1, chunk2, ...]}

    Note over Pipeline: Phase 3: Embedding
    Pipeline->>Embedder: run(documents)
    Embedder->>Embedder: encode_batch(texts)
    Embedder-->>Pipeline: {documents: [doc_with_embedding, ...]}

    Note over Pipeline: Phase 4: Storage
    Pipeline->>Writer: run(documents)
    Writer->>DocumentStore: write_documents(documents)
    DocumentStore-->>Writer: {written: 100}
    Writer-->>Pipeline: {documents_written: 100}

    Pipeline-->>User: {status: "success", documents_written: 100}
```

### 2.2 Multi-Format Document Ingestion

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant FileTypeRouter
    participant PDFConverter
    participant DOCXConverter
    participant HTMLConverter
    participant Joiner
    participant Splitter
    participant Embedder
    participant Writer

    User->>Pipeline: run(sources=[pdf, docx, html])

    Pipeline->>FileTypeRouter: run(sources)
    FileTypeRouter-->>Pipeline: route to converters

    par PDF Branch
        Pipeline->>PDFConverter: run(pdf_files)
        PDFConverter-->>Pipeline: pdf_documents
    and DOCX Branch
        Pipeline->>DOCXConverter: run(docx_files)
        DOCXConverter-->>Pipeline: docx_documents
    and HTML Branch
        Pipeline->>HTMLConverter: run(html_files)
        HTMLConverter-->>Pipeline: html_documents
    end

    Pipeline->>Joiner: run(all_documents)
    Joiner-->>Pipeline: merged_documents

    Pipeline->>Splitter: run(merged_documents)
    Splitter-->>Pipeline: chunks

    Pipeline->>Embedder: run(chunks)
    Embedder-->>Pipeline: embedded_chunks

    Pipeline->>Writer: run(embedded_chunks)
    Writer-->>Pipeline: success

    Pipeline-->>User: ingestion_complete
```

---

## 3. Chat Pipeline

### 3.1 Basic Chat Flow

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant ChatPromptBuilder
    participant ChatGenerator
    participant LLMProvider

    User->>Pipeline: run(messages=[ChatMessage.from_user("Hello")])

    Pipeline->>ChatPromptBuilder: run(messages, system_prompt)
    ChatPromptBuilder->>ChatPromptBuilder: build_messages()
    ChatPromptBuilder-->>Pipeline: {prompt: [system_msg, user_msg]}

    Pipeline->>ChatGenerator: run(messages)
    ChatGenerator->>LLMProvider: chat_completion(messages)
    LLMProvider-->>ChatGenerator: response
    ChatGenerator-->>Pipeline: {replies: [ChatMessage.from_assistant("Hi!")]}

    Pipeline-->>User: {replies: ["Hi! How can I help?"]}
```

### 3.2 Chat với RAG Context

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Embedder
    participant Retriever
    participant ChatPromptBuilder
    participant ChatGenerator

    User->>Pipeline: run(query, chat_history)

    Note over Pipeline: Retrieve relevant context
    Pipeline->>Embedder: run(query)
    Embedder-->>Pipeline: query_embedding

    Pipeline->>Retriever: run(query_embedding)
    Retriever-->>Pipeline: relevant_documents

    Note over Pipeline: Build chat with context
    Pipeline->>ChatPromptBuilder: run(query, documents, history)
    ChatPromptBuilder-->>Pipeline: messages_with_context

    Pipeline->>ChatGenerator: run(messages)
    ChatGenerator-->>Pipeline: assistant_response

    Pipeline-->>User: response_with_sources
```

---

## 4. Agent Workflow

### 4.1 Agent với Tool Calling

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant ChatGenerator
    participant LLM
    participant ToolExecutor
    participant SearchTool
    participant CalculatorTool

    User->>Agent: run("What's the weather in Tokyo and convert 100 USD to JPY?")

    loop Agent Loop (max_steps)
        Agent->>ChatGenerator: run(messages, tools)
        ChatGenerator->>LLM: chat_completion(messages, tools)
        LLM-->>ChatGenerator: response

        alt Response contains tool_calls
            ChatGenerator-->>Agent: {tool_calls: [...]}

            par Execute Tools
                Agent->>ToolExecutor: execute(weather_tool)
                ToolExecutor->>SearchTool: run(query="weather Tokyo")
                SearchTool-->>ToolExecutor: "25°C, sunny"
                ToolExecutor-->>Agent: tool_result_1
            and
                Agent->>ToolExecutor: execute(currency_tool)
                ToolExecutor->>CalculatorTool: run(100, "USD", "JPY")
                CalculatorTool-->>ToolExecutor: "15,000 JPY"
                ToolExecutor-->>Agent: tool_result_2
            end

            Agent->>Agent: update_state(tool_results)

        else Response is final answer
            ChatGenerator-->>Agent: {text: "The weather in Tokyo is..."}
            Agent-->>User: final_answer
        end
    end
```

### 4.2 Agent State Management

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant State
    participant Generator
    participant Tools

    User->>Agent: run(initial_message)

    Agent->>State: initialize()
    State-->>Agent: {messages: [], iteration: 0}

    loop Until done or max_steps
        Agent->>State: get_messages()
        State-->>Agent: current_messages

        Agent->>Generator: run(messages, tools)
        Generator-->>Agent: response

        alt Has tool_calls
            Agent->>Tools: execute(tool_calls)
            Tools-->>Agent: tool_results

            Agent->>State: add_message(assistant_response)
            Agent->>State: add_message(tool_results)
            Agent->>State: increment_iteration()
        else Final response
            Agent->>State: add_message(final_answer)
            Agent->>State: set_done()
        end
    end

    Agent->>State: get_final_state()
    State-->>Agent: final_state
    Agent-->>User: {answer, state}
```

---

## 5. Component Lifecycle

### 5.1 Component Initialization & Execution

```mermaid
sequenceDiagram
    participant Pipeline
    participant Component
    participant Model
    participant Cache

    Note over Pipeline,Component: Phase 1: Initialization
    Pipeline->>Component: __init__(config)
    Component->>Component: store_config()
    Component-->>Pipeline: component_instance

    Note over Pipeline,Component: Phase 2: Warm Up
    Pipeline->>Component: warm_up()
    Component->>Model: load_model()
    Model-->>Component: model_loaded
    Component->>Cache: initialize_cache()
    Cache-->>Component: cache_ready
    Component-->>Pipeline: ready

    Note over Pipeline,Component: Phase 3: Execution
    Pipeline->>Component: run(input_data)
    Component->>Component: validate_input()
    Component->>Model: process(data)
    Model-->>Component: result
    Component->>Component: format_output()
    Component-->>Pipeline: {output: result}
```

### 5.2 Pipeline Assembly & Validation

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Graph
    participant ComponentA
    participant ComponentB

    User->>Pipeline: Pipeline()
    Pipeline->>Graph: create_empty_graph()

    User->>Pipeline: add_component("a", ComponentA)
    Pipeline->>Graph: add_node("a", ComponentA)
    Pipeline->>Pipeline: register_sockets(ComponentA)

    User->>Pipeline: add_component("b", ComponentB)
    Pipeline->>Graph: add_node("b", ComponentB)
    Pipeline->>Pipeline: register_sockets(ComponentB)

    User->>Pipeline: connect("a.output", "b.input")
    Pipeline->>Pipeline: validate_socket_types()
    Pipeline->>Graph: add_edge("a", "b")

    User->>Pipeline: run(data)
    Pipeline->>Graph: validate_dag()
    Graph-->>Pipeline: valid
    Pipeline->>Graph: topological_sort()
    Graph-->>Pipeline: [ComponentA, ComponentB]

    Pipeline->>ComponentA: run(input)
    ComponentA-->>Pipeline: output_a

    Pipeline->>ComponentB: run(output_a)
    ComponentB-->>Pipeline: output_b

    Pipeline-->>User: final_output
```

---

## 6. Serialization Flow

### 6.1 Pipeline Serialization

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Components
    participant YAMLSerializer
    participant FileSystem

    Note over User,FileSystem: Serialize
    User->>Pipeline: to_dict()

    loop For each component
        Pipeline->>Components: component.to_dict()
        Components-->>Pipeline: component_dict
    end

    Pipeline->>Pipeline: serialize_connections()
    Pipeline-->>User: pipeline_dict

    User->>YAMLSerializer: dump(pipeline_dict)
    YAMLSerializer->>FileSystem: write("pipeline.yaml")

    Note over User,FileSystem: Deserialize
    User->>FileSystem: read("pipeline.yaml")
    FileSystem-->>User: yaml_content

    User->>YAMLSerializer: load(yaml_content)
    YAMLSerializer-->>User: pipeline_dict

    User->>Pipeline: from_dict(pipeline_dict)

    loop For each component
        Pipeline->>Components: Component.from_dict(config)
        Components-->>Pipeline: component_instance
    end

    Pipeline->>Pipeline: restore_connections()
    Pipeline-->>User: pipeline_instance
```

---

## 7. Tracing Flow

### 7.1 OpenTelemetry Tracing

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Tracer
    participant Component
    participant Exporter

    User->>Pipeline: run(data)

    Pipeline->>Tracer: start_span("pipeline.run")
    Tracer-->>Pipeline: pipeline_span

    loop For each component
        Pipeline->>Tracer: start_span("component.run")
        Tracer-->>Pipeline: component_span

        Pipeline->>Component: run(input)
        Component-->>Pipeline: output

        Pipeline->>Tracer: set_attributes(input, output)
        Pipeline->>Tracer: end_span(component_span)
    end

    Pipeline->>Tracer: end_span(pipeline_span)

    Tracer->>Exporter: export_spans()
    Exporter-->>Tracer: success

    Pipeline-->>User: result
```

---

## 8. Error Handling Flow

### 8.1 Pipeline Error Recovery

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant ComponentA
    participant ComponentB
    participant ErrorHandler

    User->>Pipeline: run(data)

    Pipeline->>ComponentA: run(input)
    ComponentA-->>Pipeline: success

    Pipeline->>ComponentB: run(data)
    ComponentB->>ComponentB: process()
    ComponentB--xPipeline: raise ComponentError

    Pipeline->>ErrorHandler: handle_error()
    ErrorHandler->>ErrorHandler: log_error()
    ErrorHandler->>ErrorHandler: cleanup()

    alt Retry enabled
        ErrorHandler->>ComponentB: run(data) [retry]
        ComponentB-->>Pipeline: success
    else No retry
        ErrorHandler-->>Pipeline: propagate_error
        Pipeline-->>User: raise PipelineError
    end
```

---

## 9. Async Pipeline Flow

### 9.1 Parallel Component Execution

```mermaid
sequenceDiagram
    participant User
    participant AsyncPipeline
    participant EmbedderA
    participant EmbedderB
    participant Joiner
    participant Generator

    User->>AsyncPipeline: run(queries)

    Note over AsyncPipeline: Parallel execution
    par Embed Query A
        AsyncPipeline->>EmbedderA: run(query_a)
        EmbedderA-->>AsyncPipeline: embedding_a
    and Embed Query B
        AsyncPipeline->>EmbedderB: run(query_b)
        EmbedderB-->>AsyncPipeline: embedding_b
    end

    Note over AsyncPipeline: Join results
    AsyncPipeline->>Joiner: run(embedding_a, embedding_b)
    Joiner-->>AsyncPipeline: combined_embeddings

    AsyncPipeline->>Generator: run(combined)
    Generator-->>AsyncPipeline: response

    AsyncPipeline-->>User: final_response
```

---

## 10. Evaluation Flow

### 10.1 RAG Evaluation Pipeline

```mermaid
sequenceDiagram
    participant User
    participant EvalPipeline
    participant RAGPipeline
    participant ContextRelevance
    participant Faithfulness
    participant AnswerCorrectness
    participant ResultAggregator

    User->>EvalPipeline: run(test_dataset)

    loop For each test case
        EvalPipeline->>RAGPipeline: run(question)
        RAGPipeline-->>EvalPipeline: {answer, contexts}

        par Evaluate metrics
            EvalPipeline->>ContextRelevance: run(question, contexts)
            ContextRelevance-->>EvalPipeline: relevance_score
        and
            EvalPipeline->>Faithfulness: run(answer, contexts)
            Faithfulness-->>EvalPipeline: faithfulness_score
        and
            EvalPipeline->>AnswerCorrectness: run(answer, ground_truth)
            AnswerCorrectness-->>EvalPipeline: correctness_score
        end

        EvalPipeline->>ResultAggregator: aggregate(scores)
    end

    ResultAggregator-->>EvalPipeline: evaluation_results
    EvalPipeline-->>User: {metrics: {...}, detailed_results: [...]}
```
