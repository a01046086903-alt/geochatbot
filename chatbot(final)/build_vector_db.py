import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_db():
    file_paths = [
        "2022_사회과_교육과정_성취기준_오세아니아.md",
        "영역별_성취수준_오세아니아.md",
        "오세아니아_단원_교과서.md",
        "오세아니아_단원_지도서.md"
    ]
    db_path = "./chroma_db"
    
    # 1. 청킹(Chunking)
    print(f"Initializing RecursiveCharacterTextSplitter (chunk_size=800, chunk_overlap=100)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    all_chunks = []
    all_metadatas = []
    
    for file_path in file_paths:
        print(f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        print(f"Splitting text into chunks for {file_path}...")
        
        # Regex to find page patterns like "p106", "p108~109"
        import re
        page_pattern = re.compile(r'p(\d+(?:~\d+)?)')
        
        sections = []
        lines = text.split('\n')
        current_section_lines = []
        current_page = None
        
        for line in lines:
            if line.strip().startswith('#'):
                if current_section_lines:
                    sections.append((current_page, '\n'.join(current_section_lines)))
                    current_section_lines = []
                match = page_pattern.search(line)
                if match:
                    current_page = match.group(1)
            current_section_lines.append(line)
            
        if current_section_lines:
            sections.append((current_page, '\n'.join(current_section_lines)))
            
        file_chunks = []
        file_metadatas = []
        
        for page, section_text in sections:
            section_chunks = splitter.split_text(section_text)
            for chunk in section_chunks:
                metadata = {
                    "source": file_path,
                    "chunk_index": len(file_chunks)
                }
                if page:
                    metadata["page"] = page
                file_chunks.append(chunk)
                file_metadatas.append(metadata)
                
        print(f"Chunks created for {file_path}: {len(file_chunks)}\n")
        all_chunks.extend(file_chunks)
        all_metadatas.extend(file_metadatas)
        
    print(f"Total chunks created across all files: {len(all_chunks)}")
    
    # 3. 벡터 DB 준비 및 2. 임베딩(Embedding)
    print("\nDeleting existing ChromaDB directory to prevent stale chunks...")
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    print("\nInitializing ChromaDB PersistentClient...")
    client = chromadb.PersistentClient(path=db_path)
    
    print("Setting up embedding function (paraphrase-multilingual-MiniLM-L12-v2)...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    collection_name = "oceania_knowledge"
    
    # Get or create collection
    print(f"Creating or getting collection: '{collection_name}'")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    # Prepare data for insertion (using source and index for unique IDs)
    ids = [f"{meta['source']}_chunk_{meta['chunk_index']}" for meta in all_metadatas]
    
    print("Adding chunks to Vector DB...")
    # Add chunks (upsert replaces if ids exist)
    collection.upsert(
        documents=all_chunks,
        ids=ids,
        metadatas=all_metadatas
    )
    
    print(f"\n[SUCCESS] Vector DB successfully built at '{db_path}'!")
    print(f"Total items in collection '{collection_name}': {collection.count()}")

if __name__ == "__main__":
    build_db()
