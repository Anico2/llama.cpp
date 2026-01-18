Folder for creating langchain projects. To better understand the inner mechanism of graph, I tried to fully comment/documents the steps into each .py. 

- [Langgraph Main Components](#langgraph-main-components)
- [Key Components of the Graph](#key-components-of-the-graph)
- [Mermaid Graph and Description](#mermaid-graph-and-description)
  - [1. Initialization and Retrieval](#1-initialization-and-retrieval)
  - [2. The Decision Hub: Grading Documents](#2-the-decision-hub-grading-documents)
  - [3. The "No Context" Verification](#3-the-no-context-verification)
  - [Summary](#summary)

# Langgraph main components
The most important conceps are( as stated in the LangGraph [graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api)):

- **State**: If defines the states in which the application can be. In this repo, they are put in `state.py`.
- **Nodes**: As the name suggest, these are the nodes of the graph and, essentially, are the working unit of the whole system. Since they are basically functions, they take an **input** (i.e. *the current system state*) and they give back an **output** (i.e. *the new system updated state*). Being a components, these are put in `components/nodes.py`.
- **Edges**: Also in this case, as the name suggets, these are the edges/link of the graph. They decide which nodes are executed next, based on the state(s) they receive. Additionally, they can be conditional or fixed.


# Key components of the graph
```
src/  
├── components/  
│   ├── chains.py        # Contains LLM Logic, i.e. grader chain, retriever chain and generator chain.
│   ├── nodes.py         # Contains Graph Nodes, i.e. the operations done for each step (see figure below).
│   └── utils.py         # Contains utilities, like models definitions.
├── graph_builder.py     # Contains all the graph workflow, i.e. defines nodes, edges and conditionals
├── main.py              # Entry point, that pars parameters and trigger execution
└── state.py             # Contains the possible states of the chain, i.e. the states that are read/modified by the nodes.
```


# Mermaid graph and description
![](mermaid_graph.png)


This flowchart represents a **Retrieval-Augmented Generation (RAG)** workflow designed with adaptive routing and fallback mechanisms. Instead of a single linear path, the system evaluates the quality of retrieved information and decides the best strategy to answer the user's request.

## 1. Initialization and Retrieval
* Start (`__start__`): The workflow initiates when a query is received.
* Retrieve (`retrieve`): The system immediately performs a retrieval step to find context relevant to the query.

## 2. The Decision Hub: Grading Documents
* Grade Documents (`grade_documents`): This is the central router. The system evaluates the retrieved documents for relevance and quality. Based on this "grade," the workflow splits into four potential paths:

    * Path A: Generate Answer (Success)
        * `generate`: If the documents are graded as relevant and sufficient, the system generates a final answer using the context.
        * `__end__`: The process completes successfully.

    * Path B: Query Refinement (Loop)
        * `transform_query`: If documents are irrelevant or insufficient, the system rewrites or optimizes the search query.
        * `retrieve` (Loop): It loops back to the retrieval step to try again with the new query.

    * Path C: Fallback Generation
        * `generate_no_context`: If retrieval fails or is not satisfactory, the system attempts to answer using the model's internal knowledge base.

    * Path D: Stop
        * `give_up`: If the documents are poor and the query cannot be answered with system knowledge, the system decides it cannot answer.
        * `__end__`: The process terminates.

## 3. The "No Context" Verification
If the system takes Path C (`generate_no_context`), it performs an extra safety check:

* Evaluate Generation (`evaluate_generation_no_context`): The system evaluates the answer generated without context to ensure it isn't a hallucination or incorrect.
    * Valid: If the answer is good, it proceeds to `__end__`.
    * Invalid: If the answer is poor, it routes to `give_up` and then terminates.

## 4. Summary
This model is an agentic workflow that prioritizes accuracy. It prefers to refine its search or admit it is not able to answer (`give_up`) rather than providing a low-quality answer based on irrelevant documents.