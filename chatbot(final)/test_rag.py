import sys
import chromadb
from chromadb.utils import embedding_functions

def query_db(query_text):
    db_path = "./chroma_db"
    
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=db_path)
    
    print("Setting up embedding function (paraphrase-multilingual-MiniLM-L12-v2)...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    collection_name = "oceania_knowledge"
    collection = client.get_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    print(f"Querying for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    print(f"\nWriting results to result.md...")
    with open("result.md", "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"Query: {query_text}\n")
        f.write("="*50 + "\n")
        f.write("--- Retrieved Documents ---\n")
        
        for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            f.write(f"\n[Result {i+1}] (Distance: {dist:.4f}, Source: {meta['source']})\n")
            f.write(doc + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    query = "오세아니아의 대표적인 기후 특징은 무엇인가요?"
    query_db(query)
