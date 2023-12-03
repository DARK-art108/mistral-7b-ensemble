## Gen AI Open Domain Chatbot

**Implementation Stack:**

Embedder : BAAI general embedding
Retrieval : FAISS Vectorstore
Generation : Mistral-7B-Instruct GPTQ model
Infrastructure : non-gpu stack, on cpu
Data: Plug Any

**What all things are Implemented as of now?**

* Utilized mistral-7b gguf quantized model from HF and BAAI/bge-small-en-v1.5 as embedding model.
* Loading LLM using Ctransformer
* Retrival QA Chain from Langchain
* Ensembling Retriever with BM25(Sparse Retriever) and Faiss Retriever(Dense Retriever)
* ChainLit used as a Chat UI wth AsyncLangchainCallbackHandler
* InMemoryCaching also enabled in both cpu and gpu.
* GPU version also available with Instruct Fine Tuning of Mistral-7b model using GPTQ.

**Setup**

* Plug your data
* Download Model
* Generate Embedding using using ```python ingest.py```
* Run the model-cpu.py, it is tested in local so using ```chainlit run model-cpu.py -w --port 8080```
* Run GPU version if you have enebled CUDA chailit is not availbale in gpu version.


**New Addons**
```(To solve Lost in Middle Problem) - 3 December 2023```

1. EmbeddingClusterFilter
2. EmbeddingRedundantFilter
3. ContextualCompressionRetriever
4. DocumentCompressorPipeline
5. LongContextReorder
6. MergerRetriever in place of EnsembleRetriver
7. ChromaDB, removed FAISS