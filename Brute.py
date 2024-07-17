def create_and_save_faiss_index(documents, embeddings_model, combined_path='faiss_and_docs.pkl', batch_size=10000):
    all_texts = []
    all_embeddings = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        embeddings = embeddings_model.embed_documents(texts)
        
        all_texts.extend(texts)
        all_embeddings.extend(embeddings)
    
    faiss_index = FAISS.from_embeddings(all_embeddings, all_texts)
    
    with open(combined_path, 'wb') as f:
        pickle.dump({'index': faiss_index, 'documents': documents}, f)
