from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

def load_faiss_index(index_path='faiss_index.bin', docs_path='documents.pkl'):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load Document objects
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)

    faiss_index = FAISS(index=index, docstore=None, docstore_index=None, embeddings=None, metadata=None, original=None)
    faiss_index.docstore = {doc.metadata['filename']: doc for doc in documents}
    return faiss_index, documents


def create_retriever(vectorstore):
    # Create retriever
    retriever = vectorstore.as_retriever()
    return retriever

def define_prompts():
    # Define prompts
    contextualize_q_system_prompt = "Given a chat history and the latest user question which might reference content in the chat history, formulate a standalone question which can be understood without the chat history. Do Not answer the question, just refromulate it if needed and otherwise return it as is."
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = "You are an assistant for question answering tasks. Use only the following pieces of retrieved context to answer the question. If the question is not related to the context then don't answer, just say that you are not sure about that. If you don't know the answer, just say that you are not sure about that in 1 or 2 lines and strictly dont exceed more than that. Question: {question} Context: {context} Answer:"
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    return contextualize_q_chain, qa_prompt

def define_chain(contextualize_q_chain, qa_prompt, retriever):
    # Define chain
    rag_chain = (
        RunnablePassthrough.assign(context=contextualized_question | retriever | format_docs) | qa_prompt | llm
    )
    return rag_chain

def contextualized_question(input):
    if input.get("chat_history"):
        return contextualize_q_chian
    else:
        return input['question']
    
def format_docs(docs):
    formatted_docs = "\n\n".join(f"{doc.page_content} (Source: {doc.metadata['source']})" if 'source' in doc.metadata else f"{doc.page_content}" for doc in docs)
    return formatted_docs

# Create retriever
retriever = create_retriever(vs)

# Define prompts
contextualize_q_chain, qa_prompt = define_prompts()

# Define chain
rag_chain = define_chain(contextualize_q_chain, qa_prompt, retriever)


print(rag_chain.invoke({"chat_history":[],"question":" ?"}))




def create_and_save_faiss_index(documents, embeddings_model, batch_size=1000, index_path='faiss_index.bin', docs_path='documents.pkl'):
    all_embeddings = []
    all_documents = []
    docstore = {}
    doc_ids = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        faiss_index = FAISS.from_documents(batch_docs, embeddings_model)
        all_embeddings.append(faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal))
        all_documents.extend(batch_docs)
        for doc in batch_docs:
            doc_id = doc.metadata['filename']
            docstore[doc_id] = doc
            doc_ids.append(doc_id)
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # Create final FAISS index
    dimension = embeddings.shape[1]
    final_index = faiss.IndexFlatL2(dimension)
    final_index.add(embeddings)

    # Save final FAISS index
    faiss.write_index(final_index, index_path)
    
    # Save Document objects and doc_ids
    with open(docs_path, 'wb') as f:
        pickle.dump((all_documents, docstore, doc_ids), f)

def load_faiss_index(index_path='faiss_index.bin', docs_path='documents.pkl'):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load Document objects, docstore, and doc_ids
    with open(docs_path, 'rb') as f:
        all_documents, docstore, doc_ids = pickle.load(f)

    def index_to_docstore_id(index):
        return doc_ids[index]

    # Create FAISS object
    faiss_index = FAISS(index=index, docstore=docstore, embedding_function=VertexAIEmbeddings().embed_documents, index_to_docstore_id=index_to_docstore_id)
    return faiss_index, all_documents
