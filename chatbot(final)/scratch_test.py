import os
import sys
import chromadb
from chromadb.utils import embedding_functions

def test_hybrid():
    db_path = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    collection = chroma_client.get_collection(
        name="oceania_knowledge", 
        embedding_function=sentence_transformer_ef
    )
    
    query = "오세아니아의 기후 특징은 무엇인가요?"
    results = collection.query(query_texts=[query], n_results=3)
    
    distances = results['distances'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    contexts = []
    print("--- 1단계 로컬 검색 결과 ---")
    for dist, doc, meta in zip(distances, documents, metadatas):
        source_name = meta.get('source', '')
        print(f"Dist: {dist:.4f}, Source: {source_name}")
        if dist < 1.5:
            if '교과서' in source_name:
                contexts.append(doc)
            elif '지도서' in source_name:
                contexts.append(doc)
                
    print(f"\n필터링 후 contexts 수: {len(contexts)}")

if __name__ == '__main__':
    test_hybrid()
