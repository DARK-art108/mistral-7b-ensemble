from langchain.embeddings import HuggingFaceEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstoredb/db_faiss'


def load_doc():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    esops_documents = text_splitter.transform_documents(documents)
    return esops_documents

def vectorize_doc(cache_embedder):
    doc = load_doc()
    vectorstore = FAISS.from_documents(doc, cache_embedder)
    vectorstore.save_local(DB_FAISS_PATH)
    bm25_retriever = BM25Retriever.from_documents(doc)
    bm25_retriever.k = 5
    return vectorstore, bm25_retriever

def load_db():
    store = LocalFileStore("./cache")
    embed_model = 'BAAI/bge-small-en-v1.5'
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
    embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model, store, namespace=embed_model)
    vectorstore, bm25_retriever = vectorize_doc(embedder)
    faiss_retriever = vectorstore.as_retriever(search_kwargs = {"k": 5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever], weights = [0.5,0.5])
    return ensemble_retriever

# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
#     esops_documents = text_splitter.transform_documents(documents)
#     store = LocalFileStore("./cache")
#     embed_model = 'BAAI/bge-small-en-v1.5'
#     embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
#     embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model, store, namespace=embed_model)
#     vectorstore = FAISS.from_documents(esops_documents, embedder)
#     vectorstore.save_local(DB_FAISS_PATH)
#     bm25_retriever = BM25Retriever.from_documents(esops_documents)
#     bm25_retriever.k = 5
#     faiss_retriever = vectorstore.as_retriever(search_kwargs = {"k": 5})
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever], weights = [0.5,0.5])
#     return ensemble_retriever

if __name__ == "__main__":                      
    load_db()  



    