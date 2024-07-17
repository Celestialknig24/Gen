from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import pickle

# Initialize Vertex AI models
embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@latest")
llm = VertexAI(model="gemini-pro", top_k=5, top_p=0.9, temperature=0.7, max_output_tokens=2048)

# Helper function to chunk documents using CharacterTextSplitter
def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        chunked_docs.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])
    return chunked_docs

def create_and_save_faiss_index(documents, embeddings_model, combined_path='faiss_and_docs.pkl'):
    chunked_docs = chunk_documents(documents)
    texts = [doc.page_content for doc in chunked_docs]
    embeddings = embeddings_model.embed_documents(texts)
    faiss_index = FAISS.from_embeddings(embeddings, texts)
    
    with open(combined_path, 'wb') as f:
        pickle.dump({'index': faiss_index, 'documents': chunked_docs}, f)

def load_faiss_index(combined_path='faiss_and_docs.pkl'):
    with open(combined_path, 'rb') as f:
        data = pickle.load(f)
    return data['index'], data['documents']

def retrieve_relevant_chunks(query, faiss_index, embeddings_model, top_k=5):
    query_embedding = embeddings_model.embed_query(query)
    D, I = faiss_index.search(query_embedding, top_k)
    retrieved_docs = [faiss_index.documents[i] for i in I[0]]
    return retrieved_docs

def generate_response(query, faiss_index, embeddings_model, llm, top_k=5):
    retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, embeddings_model, top_k)
    input_text = query + " ".join([doc.page_content for doc in retrieved_chunks])
    response = llm.generate(input_text)
    return response

# Example usage
documents = [Document(page_content="Your large text document goes here...")] * 10  # Replace with actual documents
create_and_save_faiss_index(documents, embeddings_model)

faiss_index, docs = load_faiss_index()

# Perform RAG
query = "What is the capital of France?"
response = generate_response(query, faiss_index, embeddings_model, llm)
print(response)





---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[13], line 1
----> 1 faiss_index = FAISS.from_embeddings(emb,txt)

File /opt/conda/lib/python3.10/site-packages/langchain_community/vectorstores/faiss.py:1006, in FAISS.from_embeddings(cls, text_embeddings, embedding, metadatas, ids, **kwargs)
    977 @classmethod
    978 def from_embeddings(
    979     cls,
   (...)
    984     **kwargs: Any,
    985 ) -> FAISS:
    986     """Construct FAISS wrapper from raw documents.
    987 
    988     This is a user friendly interface that:
   (...)
   1004             faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
   1005     """
-> 1006     texts, embeddings = zip(*text_embeddings)
   1007     return cls.__from(
   1008         list(texts),
   1009         list(embeddings),
   (...)
   1013         **kwargs,
   1014     )

ValueError: too many values to unpack (expected 2)
