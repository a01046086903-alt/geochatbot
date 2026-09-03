import os
import urllib.request
import urllib.parse
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 환경 변수로 API 키 불러오기
# ==========================================
# 실행 전 터미널에서 다음 명령어로 환경변수를 설정해주세요. (윈도우 PowerShell 기준)
# $env:NAVER_CLIENT_ID="Z3ctnxISEw4WbUOKGxP7"
# $env:NAVER_CLIENT_SECRET="B_RyJtcnoJ"
# $env:GEMINI_API_KEY="AQ.Ab8RN6IwVBeI1lO0j0SrT_CFwHqJu_txhwG7ORomkc5ex91afg"

NAVER_CLIENT_ID = "Z3ctnxISEw4WbUOKGxP7"
NAVER_CLIENT_SECRET = "B_RyJtcnoJ"
GEMINI_API_KEY = "AQ.Ab8RN6IwVBeI1lO0j0SrT_CFwHqJu_txhwG7ORomkc5ex91afg"

db_path = "./chroma_db"
try:
    chroma_client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name="oceania_knowledge", embedding_function=sentence_transformer_ef)
except Exception as e:
    print(f"[오류] 벡터 DB 초기화 실패: {e}")
    collection = None

def search_chromadb(query, threshold=0.4):
    """1단계: 로컬 벡터 DB (ChromaDB) 검색"""
    if not collection:
        print("[오류] 벡터 DB 컬렉션이 없습니다.")
        return []
    
    try:
        results = collection.query(query_texts=[query], n_results=3)
    except Exception as e:
        print(f"[오류] DB 검색 중 에러 발생: {e}")
        return []
        
    contexts = []
    distances = results['distances'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for dist, doc, meta in zip(distances, documents, metadatas):
        # L2 Distance가 threshold보다 낮아야(가까워야) 유의미한 정보로 판단
        if dist < threshold:
            source_name = meta['source'].replace('.md', '')
            contexts.append({"doc": doc, "source": source_name})
            
    return contexts

def search_naver_encyc(query):
    """2단계: 네이버 지식백과 API 검색"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("[알림] 네이버 API 키가 설정되지 않아 지식백과 검색을 건너뜁니다.")
        return []
        
    encText = urllib.parse.quote(query)
    url = "https://openapi.naver.com/v1/search/encyc.json?query=" + encText + "&display=3"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
    
    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read()
            data = json.loads(response_body.decode('utf-8'))
            items = data.get("items", [])
            contexts = []
            for item in items:
                # 불필요한 HTML 태그(<b>, </b> 등) 제거
                desc = item.get("description", "").replace("<b>", "").replace("</b>", "")
                title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                contexts.append({"doc": f"{title}: {desc}", "source": "네이버 지식백과"})
            return contexts
    except Exception as e:
        print(f"[오류] 네이버 API 호출 중 문제가 발생했습니다: {e}")
        
    return []

def generate_answer(query, contexts):
    """3단계: Gemini 모델로 답변 생성 (시스템 프롬프트 적용)"""
    if not GEMINI_API_KEY:
        return "[오류] GEMINI_API_KEY가 설정되지 않았습니다."
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 핵심 조건 1, 3 반영 (페르소나, 출처 표기, 환각 방지)
    system_instruction = """당신은 중학교 1학년 학생들에게 다정하고 친근하게 사회를 가르치는 '사람 형태의 사회 선생님'입니다. (동물 캐릭터가 아닙니다).
학생들을 존중하고 따뜻하게 격려하는 '~해요', '~알아볼까요?' 말투를 사용하세요.

[답변 및 출처 제약 조건]
1. 제공된 [지식]이 있다면 반드시 그 내용을 기반으로 답변해야 하며, 답변 맨 끝에 제공된 'Source' 정보를 활용해 무조건 '[출처: OOO]' 형식으로 기재하세요. (예: [출처: 오세아니아_단원_교과서], [출처: 네이버 지식백과]).
2. 만약 제공된 [지식]의 텍스트(Content) 내부에 페이지 번호(예: p106, 106쪽 등)가 적혀있다면, 출처 표기 시 페이지 번호를 함께 적어주세요. (예: [출처: 오세아니아_단원_지도서, p106]). 단, 텍스트에 페이지 번호가 명시되어 있지 않다면 절대 지어내지 마세요.
3. 제공된 [지식]이 있을 경우 절대 "[선생님이 가진 추가 지식으로 답변해 줄게요!]"라는 문구를 사용하지 마세요.
4. 제공된 [지식]이 비어있을 때만, 학생의 질문에 선생님의 자체 지식으로 답변하세요. 이때는 답변의 맨 앞에 반드시 "[선생님이 가진 추가 지식으로 답변해 줄게요!]" 라는 안내 문구를 고정으로 출력한 후 답변해야 합니다. 자체 지식을 사용할 때는 출처 표기를 하지 않아도 됩니다.
"""

    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
    
    if contexts:
        # 컨텍스트 조립
        context_str = "\n\n".join([f"Source: {c['source']}\nContent: {c['doc']}" for c in contexts])
        prompt = f"다음 지식을 바탕으로 학생의 질문에 답해주세요.\n\n[지식]\n{context_str}\n\n[학생의 질문]\n{query}"
    else:
        # 핵심 조건 2 반영 (자체 지식 사용)
        prompt = f"[학생의 질문]\n{query}"
        
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[오류] 제미나이 답변 생성 중 문제가 발생했습니다: {e}"

def run_hybrid_rag(query):
    print("\n" + "="*50)
    print(f"🙋 학생 질문: {query}")
    print("="*50)
    
    # 1단계
    print("🔍 1단계: 로컬 벡터 DB(ChromaDB) 검색 중...")
    contexts = search_chromadb(query)
    
    if contexts:
        print("✅ 로컬 벡터 DB에서 관련 문맥을 찾았습니다!")
    else:
        # 2단계
        print("⚠️ 로컬 벡터 DB에 관련 정보가 부족합니다.")
        print("🔍 2단계: 네이버 지식백과 API 검색 중...")
        contexts = search_naver_encyc(query)
        if contexts:
            print("✅ 네이버 지식백과에서 관련 문맥을 찾았습니다!")
        else:
            # 3단계
            print("⚠️ 네이버 지식백과에도 정보가 없습니다.")
            print("🔍 3단계: Gemini 자체 지식 활용 준비 중...")
            contexts = []
            
    print("✨ 답변 생성 중...\n")
    answer = generate_answer(query, contexts)
    
    print("👩‍🏫 [선생님의 답변]")
    print(answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    # 윈도우 터미널 인코딩 문제 방지
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 테스트를 위한 질문 리스트
    test_queries = [
        "오세아니아의 대표적인 기후 특징은 무엇인가요?", # 로컬 DB 탐색 예상
        "오세아니아라는 이름의 유래가 궁금해요.", # 로컬 DB에 없을 확률이 높아 네이버/Gemini 탐색 예상
        "초신성 폭발이 뭐야?" # 사회교과와 무관하여 Gemini 자체 지식 탐색 예상
    ]
    
    print("🚀 하이브리드 RAG 파이프라인 테스트를 시작합니다.")
    for q in test_queries:
        run_hybrid_rag(q)
