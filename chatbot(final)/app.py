import os
import json
import threading
import urllib.parse
import urllib.request
from datetime import datetime

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ----------------------------------------------------------------------------
# 환경 변수 로드
# ----------------------------------------------------------------------------
if os.path.exists("api키.env"):
    load_dotenv("api키.env")
else:
    load_dotenv()

# 기존에 하드코딩되었던 키들이 .env에 있다고 가정하거나, 없으면 하드코딩 값을 fallback으로 사용합니다.
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "Z3ctnxISEw4WbUOKGxP7")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "B_RyJtcnoJ")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6IwVBeI1lO0j0SrT_CFwHqJu_txhwG7ORomkc5ex91afg")

# ----------------------------------------------------------------------------
# 전역 리소스 초기화 (Streamlit Cache 활용)
# ----------------------------------------------------------------------------
@st.cache_resource
def init_resources(gemini_key, naver_id, naver_secret):
    resources = {}
    
    # 1. Gemini 설정
    if gemini_key:
        genai.configure(api_key=gemini_key)
        resources['gemini_model'] = genai.GenerativeModel('gemini-2.5-flash')
    
    # 2. ChromaDB 설정
    try:
        db_path = "./chroma_db"
        chroma_client = chromadb.PersistentClient(path=db_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        resources['chroma_collection'] = chroma_client.get_collection(
            name="oceania_knowledge", 
            embedding_function=sentence_transformer_ef
        )
    except Exception as e:
        print(f"ChromaDB 초기화 오류: {e}")
        resources['chroma_collection'] = None
        
    # 3. 구글 스프레드시트 설정
    try:
        if os.path.exists('google_creds.json'):
            # gspread 5.0+ 최신 인증 방식 사용
            gclient = gspread.service_account(filename='google_creds.json')
            resources['gsheet'] = gclient.open("ChatBot_Logs").sheet1
            print("[성공] Google Sheets 연동 완료!")
        else:
            print("[알림] google_creds.json 파일이 존재하지 않습니다.")
            resources['gsheet'] = None
    except Exception as e:
        print(f"[오류] Google Sheets 초기화 실패: {e}")
        resources['gsheet'] = None
        
    return resources

res = init_resources(GEMINI_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
model = res.get('gemini_model')
collection = res.get('chroma_collection')
gsheet = res.get('gsheet')

# ----------------------------------------------------------------------------
# 기능 함수 정의
# ----------------------------------------------------------------------------

def log_to_sheet_async(timestamp, student_id, student_name, user_query, bot_response, source):
    """구글 시트에 로그를 비동기적으로 기록하는 함수"""
    def log_task():
        if gsheet:
            try:
                gsheet.append_row([timestamp, student_id, student_name, user_query, bot_response, source], value_input_option="RAW")
            except Exception as e:
                print(f"시트 기록 실패: {e}")
    
    thread = threading.Thread(target=log_task)
    thread.start()

def filter_relevant_contexts(query, contexts):
    """Gemini를 사용하여 검색된 컨텍스트들 중 질문과 실제로 관련 있는 것만 필터링"""
    if not model or not contexts:
        return contexts
        
    try:
        # 각 컨텍스트에 임시 ID 부여하여 전달
        context_items = []
        for idx, c in enumerate(contexts):
            context_items.append(f"ID: {idx}\n출처: {c['source']}\n내용: {c['doc']}")
            
        context_str = "\n\n".join(context_items)
        
        prompt = f"""[학생의 질문]에 대답하는 데 직접적인 도움이 되는 관련 정보를 담고 있는 [후보 문서]들의 ID를 골라주세요.
질문에 답하는 데 필요한 핵심 사실이나 설명이 포함되어 있다면 관련이 있는 것입니다.
반면, 질문과 전혀 무관하거나 단순한 대단원/소단원 제목, 목차 수준의 정보라면 관련이 없으므로 제외해야 합니다.

[학생의 질문]: {query}

[후보 문서 목록]:
{context_str}

출력 형식: 관련이 있는 문서의 ID들을 쉼표로 구분하여 출력하세요. (예: 0, 2)
만약 모든 문서가 질문과 전혀 관련이 없고 엉뚱한 내용이라면 반드시 'NONE'이라고만 출력하세요.
다른 설명은 절대 하지 마세요."""
        
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        
        if "NONE" in result:
            return []
            
        relevant_indices = []
        for word in result.replace(",", " ").split():
            if word.isdigit():
                idx = int(word)
                if 0 <= idx < len(contexts):
                    relevant_indices.append(idx)
                    
        return [contexts[i] for i in relevant_indices]
    except Exception as e:
        print(f"컨텍스트 필터링 오류: {e}")
        return contexts

def search_hybrid(query):
    """3단계 하이브리드 검색 로직"""
    # 1단계: 로컬 교과서/지도서 검색 (ChromaDB)
    if collection:
        try:
            results = collection.query(query_texts=[query], n_results=3)
            contexts = []
            distances = results['distances'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # 유사도 임계값 0.5 적용 (L2 거리 기준, 작을수록 유사함. 임계값을 완화하고 LLM으로 관련성 2차 검증 수행)
            for dist, doc, meta in zip(distances, documents, metadatas):
                if dist < 0.5:
                    source_name = meta.get('source', '')
                    page = meta.get('page', '')
                    page_str = f" p.{page}" if page else ""
                    
                    if '교과서' in source_name:
                        display_source = f"교과서 {page_str}".strip()
                    elif '지도서' in source_name:
                        display_source = f"지도서 {page_str}".strip()
                    else:
                        continue
                        
                    contexts.append({"doc": doc, "source": display_source})
            
            # Gemini를 통한 2차 관련성 검증
            if contexts:
                filtered = filter_relevant_contexts(query, contexts)
                if filtered:
                    return filtered, "Local"
        except Exception as e:
            print(f"ChromaDB 검색 오류: {e}")

    # 2단계: 네이버 지식백과 API 검색 (Fallback 1)
    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
        encText = urllib.parse.quote(query)
        url = "https://openapi.naver.com/v1/search/encyc.json?query=" + encText + "&display=3"
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        req.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        
        try:
            response = urllib.request.urlopen(req)
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                items = data.get("items", [])
                contexts = []
                for item in items:
                    desc = item.get("description", "").replace("<b>", "").replace("</b>", "")
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    link = item.get("link", "")
                    contexts.append({"doc": f"{title}: {desc}", "source": f"네이버 지식백과 - {title}", "link": link})
                
                # 네이버 검색 결과도 Gemini를 통한 관련성 검증 적용
                if contexts:
                    filtered = filter_relevant_contexts(query, contexts)
                    if filtered:
                        return filtered, "Naver"
        except Exception as e:
            print(f"네이버 API 검색 오류: {e}")

    # 3단계: LLM 내재 지식 활용 (Fallback 2)
    return [], "Gemini"

def get_system_prompt(section, search_stage):
    """단원 및 검색 단계에 따른 시스템 프롬프트 반환 (rag_pipeline.py 기반)"""
    
    # 기본 페르소나 설정
    base_persona = """당신은 중학교 1학년 학생들에게 다정하고 친근하게 사회를 가르치는 '사회 선생님'입니다.
학생들을 존중하고 따뜻하게 격려하는 '~해요', '~알아볼까요?' 말투를 사용하세요. 
전문 용어가 나오면 중학교 1학년 눈높이에 맞춰 아주 쉽게 풀어서 설명해주세요."""

    # 단원별 기능 설정
    if search_stage == "Gemini":
        section_prompt = f"주요 역할: 학생의 질문에 자체 지식을 활용하여 답변하되, 질문이 오세아니아 사회 교과 내용이나 현재 단원({section})과 무관한 경우 절대 답변을 제공하지 말고 학습 내용으로 유도하는 것입니다."
    else:
        if section == "6-1. 오세아니아의 지리적 특성과 자원 수출":
            section_prompt = "주요 역할: 이 단원에서는 '정보 탐색 중심 챗봇'으로 활동합니다. 학생이 오세아니아의 지리적 특성과 자원 수출에 대해 깊이 있게 정보를 탐색하고 이해할 수 있도록, 핵심적인 사실을 정확하고 풍부하게 제공하며 관련 지식을 잘 찾아주는 역할을 하세요."
        elif section == "6-2. 태평양 지역의 환경 문제와 해결 방안":
            section_prompt = "주요 역할: 이 단원에서는 '발표 자료 준비 도우미 챗봇'으로 활동합니다. 학생이 태평양 지역의 환경 문제와 해결 방안에 대한 발표 자료를 만들 때 활용하기 좋은 유익한 자료를 적극적으로 추천해주고, 발표 내용 구성에 도움이 되는 참신한 아이디어를 풍부하게 제공하세요."
        else: # 6-3. 극지방의 중요성과 지역 개발
            section_prompt = "주요 역할: 이 단원에서는 '역할극 대본 보조 및 논리 검증 챗봇'으로 활동합니다. 극지방 관련 역할극 대본을 만들 때 상황 설정과 대사 작성을 적극적으로 돕고, 학생이 작성한 주장하는 글을 분석하여 논리적인 모순이나 근거의 부족함이 없는지 예리하게 검증하고 피드백을 제공하세요."

    # 검색 단계별 제약 조건
    stage_prompt = """[답변 및 출처 제약 조건]
1. 제공된 [지식]이 있다면 반드시 그 내용을 기반으로 답변해야 합니다. 이때 답변 텍스트 내(답변 맨 하단 등)에는 절대 '출처'나 '사이트 주소(URL)'를 직접 적지 마세요. (출처 표기는 시스템 UI가 별도로 처리합니다).
2. 만약 제공된 [지식]의 텍스트(Content) 내부에 페이지 번호(예: p106, 106쪽 등)가 적혀있다면, 출처 표기 시 페이지 번호를 함께 적어주세요. 단, 텍스트에 페이지 번호가 명시되어 있지 않다면 절대 지어내지 마세요.
3. 제공된 [지식]이 있을 경우 절대 "[선생님이 가진 추가 지식으로 답변해 줄게요!]"라는 문구를 사용하지 마세요.
4. 제공된 [지식]이 비어있을 때만 선생님의 자체 지식으로 답변합니다. 이때는 답변 맨 앞에 반드시 "[선생님이 가진 추가 지식으로 답변해 줄게요!]" 라는 안내 문구를 출력하세요. 자체 지식으로 답변할 때에는 답변 텍스트 맨 마지막 줄에 반드시 '<source>출처기관명|URL</source>' 형식으로 참고한 실제 공신력 있는 웹사이트 주소를 1개만 적어주세요. (예: <source>유엔환경계획|https://www.unep.org</source>). 주의: 이 <source> 태그를 절대 마크다운 코드 블록(```)으로 감싸지 말고 그냥 생텍스트로 적으세요. 절대로 '네이버 지식백과'나 '나무위키'를 출처로 적지 마세요.
5. [필수 거절 제약] 학생이 사회 교과 및 현재 학습 단원과 아예 상관없는 엉뚱한 질문을 할 경우에는 절대 지식이나 정답을 알려주지 마세요. "선생님은 사회 수업을 위한 챗봇이에요."라며 정중하게 거절한 뒤, 학습 내용에 다시 집중할 수 있도록 현재 단원과 관련된 흥미로운 추천 질문을 1~2개 직접 제시해주세요."""

    return f"{base_persona}\n\n{section_prompt}\n\n{stage_prompt}"

@st.cache_data
def get_initial_questions(section):
    """성취기준을 바탕으로 3개의 탐구 질문 생성"""
    if not model:
        return ["오세아니아의 대표적인 기후는 무엇인가요?", "태평양의 주요 환경 문제는 어떤 것들이 있나요?", "오세아니아 사람들은 어떤 집에 살고 있나요?"]
        
    try:
        with open("2022_사회과_교육과정_성취기준_오세아니아.md", "r", encoding="utf-8") as f:
            content = f.read()
            
        prompt = f"""다음은 중학교 사회과 오세아니아 단원의 성취기준입니다.
{content}

현재 학습 중인 세부 주제는 '{section}' 입니다.
이 성취기준과 현재 학습 주제를 바탕으로 중학교 1학년 학생이 호기심을 가질 만한 흥미로운 탐구 질문 3가지를 생성해주세요.
반드시 질문은 핵심만 담아 아주 짧고 간결하게(20자 이내) 작성하세요.
반드시 각 질문은 줄바꿈으로 구분된 텍스트로만 출력하세요. (예: 1. 오세아니아 기후는?)"""

        response = model.generate_content(prompt)
        questions = [q.strip().lstrip("1234567890. ") for q in response.text.strip().split('\n') if q.strip()]
        return questions[:3] if len(questions) >= 3 else ["오세아니아의 기후는 어떤가요?", "태평양의 쓰레기 섬은 왜 생겼나요?", "원주민들은 어떻게 살았나요?"]
    except Exception as e:
        print(f"초기 질문 생성 오류: {e}")
        return ["오세아니아의 기후 특징은 무엇인가요?", "태평양의 환경 문제를 어떻게 해결할 수 있을까요?", "오세아니아의 독특한 동물은 무엇이 있나요?"]

@st.cache_data
def get_followup_questions(user_query, bot_response):
    """학생의 최근 질문과 답변을 바탕으로 3개의 후속 질문 생성"""
    if not model:
        return ["기후 변화의 영향은?", "오세아니아의 자원은?", "태평양의 섬들의 운명은?"]
    try:
        prompt = f"""중학교 1학년 학생이 오세아니아 사회 학습 중 다음과 같은 대화를 나눴습니다.
학생 질문: {user_query}
선생님 답변: {bot_response}

이 대화 내용에 이어 학생이 스스로 탐구를 심화하거나 궁금해할 만한 '꼬리 질문(후속 질문)' 3가지를 만들어주세요.
반드시 질문은 핵심만 담아 아주 짧고 간결하게(20자 이내) 작성하세요.
반드시 각 질문은 줄바꿈으로 구분된 텍스트로만 출력하세요. (예: 1. ~~~?)"""
        response = model.generate_content(prompt)
        questions = [q.strip().lstrip("1234567890. ") for q in response.text.strip().split('\n') if q.strip()]
        return questions[:3] if len(questions) >= 3 else ["기후 변화의 영향은?", "오세아니아의 자원은?", "태평양의 섬들의 운명은?"]
    except Exception as e:
        print(f"후속 질문 생성 오류: {e}")
        return ["기후 변화의 영향은?", "오세아니아의 자원은?", "태평양의 섬들의 운명은?"]

@st.cache_data
def extract_key_sentence(doc, answer):
    """문맥(doc) 중에서 답변(answer)에 직접적으로 사용되었거나 이를 뒷받침하는 핵심 내용을 완벽한 문장으로 추출"""
    if not model:
        return doc.split('.')[0] + "." if '.' in doc else doc[:60]
    try:
        prompt = f"""다음 [문맥] 중에서 [답변]의 근거가 되는 핵심 내용을 찾아서, 자연스럽고 완벽한 하나의 문장으로 정리하여 출력하세요.
단순히 문맥을 복사하지 말고, 답변을 잘 뒷받침하는 핵심 내용이 되도록 완결된 문장으로 다듬어주세요.

[문맥]: {doc}
[답변]: {answer}

출력 형식: 반드시 정리된 문장 1개만 출력하세요. 다른 부가적인 설명은 전혀 하지 마세요."""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"핵심 문장 추출 오류: {e}")
        return doc.split('.')[0] + "." if '.' in doc else doc[:60]

def render_followup_questions(questions, turn_id):
    """후속 질문 3개를 3개의 다른 색상 컬럼 버튼으로 렌더링"""
    if not questions:
        return
    st.markdown("<p style='font-size: 14px; color: gray; margin-top: 15px; font-weight: bold;'>💡 관련해서 이런 질문도 추천해요:</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    btn_types = ["primary", "secondary", "primary"]
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"followup_{i}_{turn_id}", type=btn_types[i % 3], use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

def render_feedback_buttons(idx):
    """답변 재요청 버튼 2개를 렌더링"""
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("💬 더 쉽게 설명해 주세요", key=f"btn_easier_{idx}", type="secondary", use_container_width=True):
            st.session_state.pending_question = "방금 전 답변이 이해하기 조금 어려워요. 중학교 1학년 눈높이에 맞춰서 더 쉽고 친절하게, 예시를 들어 다시 설명해 주실 수 있나요?"
            st.rerun()
    with btn_col2:
        if st.button("⚠️ 답변 내용이 잘못되었어요", key=f"btn_wrong_{idx}", type="secondary", use_container_width=True):
            st.session_state.pending_question = "방금 전 답변에 잘못된 내용이나 오류가 있는 것 같아요. 내용을 꼼꼼히 다시 확인해서 정확한 사실로 정정하여 다시 설명해 주실 수 있나요?"
            st.rerun()

def generate_quiz(chat_log):
    """대화 로그를 바탕으로 퀴즈 생성"""
    if not model:
        return None
        
    prompt = f"""다음은 선생님과 학생의 대화 내용입니다:
{chat_log}

이 내용을 바탕으로 학생이 복습할 수 있는 중학교 1학년 수준의 퀴즈(객관식 4지 또는 5지 선다형) 1문항을 만들어주세요.
반드시 아래 형식에 맞춰서 출력해주세요.

[문제]
(여기에 문제의 질문만 작성하세요. 친절한 선생님의 말투로 질문하세요.)

[보기]
(여기에 4개 또는 5개의 보기를 작성하되, 번호는 반드시 원문자(①, ②, ③, ④, ⑤)를 사용하고 각 보기를 줄바꿈으로 구분해서 세로로 배치되게 하세요. 예: 
① OOO
② OOO
③ OOO
④ OOO)

[정답 및 해설]
(여기에 정답과 해설을 작성하되, 정답과 해설 사이에 줄바꿈을 두어 문단을 명확히 분리하여 가독성을 높여주세요.)"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return None

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="AI 사회 선생님 ", page_icon="👨‍🏫", layout="wide")

st.markdown("""
<style>
/* 사이드바 라디오 버튼 박스형 디자인 */
div[role="radiogroup"] > label {
    background-color: #F8F9FA;
    padding: 12px 15px;
    border-radius: 10px;
    border: 2px solid #E9ECEF;
    margin-bottom: 8px;
    cursor: pointer;
}
div[role="radiogroup"] > label:hover {
    background-color: #E8F4F8;
    border-color: #BFE0EC;
}
div[role="radiogroup"] > label[data-checked="true"], 
div[role="radiogroup"] > label[aria-checked="true"] {
    background-color: #E8F4F8 !important;
    border-color: #1E6091 !important;
}
div[role="radiogroup"] label p {
    font-size: 16px !important;
    font-weight: bold;
    margin: 0;
}

/* 추천 질문 버튼 색상 다르게 적용 (1, 2, 3번째 컬럼) */
div[data-testid="stColumn"]:nth-of-type(1) button { background-color: #E8F4F8 !important; border: 1px solid #BFE0EC !important; color: #1E6091 !important; font-weight: 600; }
div[data-testid="stColumn"]:nth-of-type(2) button { background-color: #E8F8F5 !important; border: 1px solid #A9DFBF !important; color: #117A65 !important; font-weight: 600; }
div[data-testid="stColumn"]:nth-of-type(3) button { background-color: #FEF9E7 !important; border: 1px solid #F9E79F !important; color: #B7950B !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("👨‍🏫 AI 사회 선생님")
st.markdown("중학교 1학년 수준에 맞게 오세아니아에 대해 친절하게 알려주는 선생님입니다. 자유롭게 질문해 보세요!")

# 세션 상태 초기화 (로그인 관련)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "student_id" not in st.session_state:
    st.session_state.student_id = ""
if "student_name" not in st.session_state:
    st.session_state.student_name = ""

# 로그인 안 한 경우 로그인 화면 표시 후 로직 중단
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 태블릿/PC 등 넓은 화면에서 폼이 너무 길어지지 않도록 중앙 배치 (1:2:1 비율)
    spacer1, main_col, spacer2 = st.columns([1, 2, 1])
    with main_col:
        st.markdown("<h3 style='text-align: center;'>👋 챗봇에 입장하기 위해<br>학번과 이름을 입력해주세요.</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login_form"):
            col1, col2 = st.columns(2)
            with col1:
                sid = st.text_input("학번 (예: 10101)")
            with col2:
                sname = st.text_input("이름 (예: 홍길동)")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("입장하기", use_container_width=True, type="primary")
            if submitted:
                if sid.strip() and sname.strip():
                    st.session_state.logged_in = True
                    st.session_state.student_id = sid.strip()
                    st.session_state.student_name = sname.strip()
                    st.rerun()
                else:
                    st.error("학번과 이름을 모두 입력해주세요!")
    st.stop() # 이후 챗봇 UI 로직은 실행하지 않음

# 사이드바: 단원 선택
with st.sidebar:
    st.header("📚 학습 단원 선택")
    section = st.radio("현재 학습 중인 단원을 선택하세요:", 
             ("6-1. 오세아니아의 지리적 특성과 자원 수출", 
              "6-2. 태평양 지역의 환경 문제와 해결 방안", 
              "6-3. 극지방의 중요성과 지역 개발"))
    st.markdown("---")
    st.info("선택한 단원에 따라 선생님의 지도 방식이 달라집니다!")
    
    try:
        with open("2022_사회과_교육과정_성취기준_오세아니아.md", "r", encoding="utf-8") as f:
            lines = f.readlines()
            # [9사...] 가 포함되고 '*'로 시작하지 않는 실제 성취기준 텍스트만 추출
            standards = [line.strip() for line in lines if '[9사' in line and not line.strip().startswith('*')]
            
        st.markdown("<h4 style='color: #1E6091; margin-bottom: 5px;'>📖 핵심 성취기준</h4>", unsafe_allow_html=True)
        for i, std in enumerate(standards):
            if i % 3 == 0:
                st.info(std)
            elif i % 3 == 1:
                st.success(std)
            else:
                st.warning(std)
        
        # 영역별 성취수준 박스 (클릭 시 툴팁처럼 나오는 Popover 활용)
        with open("영역별_성취수준_오세아니아.md", "r", encoding="utf-8") as f:
            levels_content = f.read()
            
        # 성취수준 내용 예쁘게 포매팅 (HTML 박스로 감싸기 위해 \n을 <br>로 변환)
        levels_content = levels_content[levels_content.find('A수준'):].replace('\n', '<br>')
        
        formatted_levels = levels_content.replace('A수준', '<div style="background-color: rgba(46, 204, 113, 0.15); padding: 15px; border-radius: 10px; margin-bottom: 15px; font-size: 14px;"><h4 style="color: #27AE60; margin-top:0; margin-bottom: 10px;">🟢 A 수준</h4>') \
                                         .replace('B수준', '</div><div style="background-color: rgba(52, 152, 219, 0.15); padding: 15px; border-radius: 10px; margin-bottom: 15px; font-size: 14px;"><h4 style="color: #2980B9; margin-top:0; margin-bottom: 10px;">🔵 B 수준</h4>') \
                                         .replace('C수준', '</div><div style="background-color: rgba(241, 196, 15, 0.15); padding: 15px; border-radius: 10px; margin-bottom: 15px; font-size: 14px;"><h4 style="color: #F39C12; margin-top:0; margin-bottom: 10px;">🟡 C 수준</h4>') \
                                         .replace('D수준', '</div><div style="background-color: rgba(230, 126, 34, 0.15); padding: 15px; border-radius: 10px; margin-bottom: 15px; font-size: 14px;"><h4 style="color: #D35400; margin-top:0; margin-bottom: 10px;">🟠 D 수준</h4>') \
                                         .replace('E수준', '</div><div style="background-color: rgba(231, 76, 60, 0.15); padding: 15px; border-radius: 10px; margin-bottom: 15px; font-size: 14px;"><h4 style="color: #C0392B; margin-top:0; margin-bottom: 10px;">🔴 E 수준</h4>') \
                                         .replace('지식･이해:', '<b style="color: #333;">🧠 지식･이해:</b>') \
                                         .replace('과정･기능:', '<br><b style="color: #333;">⚙️ 과정･기능:</b>') \
                                         .replace('가치･태도:', '<br><b style="color: #333;">❤️ 가치･태도:</b>') + "</div>"
            
        with st.popover("📊 영역별 성취수준 보기 (A~E)", use_container_width=True):
            st.markdown(formatted_levels, unsafe_allow_html=True)
            
    except Exception as e:
        print(f"사이드바 UI 로드 오류: {e}")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# 단원이 변경되었을 때 초기 질문 및 대화 기록 갱신
if "current_section" not in st.session_state or st.session_state.current_section != section:
    st.session_state.current_section = section
    st.session_state.initial_questions = get_initial_questions(section)
    st.session_state.messages = []
    st.session_state.query_count = 0
    if "followup_questions" in st.session_state:
        del st.session_state["followup_questions"]
    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]

# 현재 사용자가 새로운 입력을 보냈는지 여부 판단 (채팅 입력창 또는 추천 질문 클릭)
has_new_input = bool(st.session_state.get('chat_input_val') or st.session_state.get('pending_question'))

# 초기 추천 질문 버튼
if not st.session_state.messages and not has_new_input:
    st.markdown("### 💡 이런 질문을 해보는 건 어떨까요?")
    st.markdown("<p style='font-size: 14px; color: gray;'>질문 버튼을 클릭하면 바로 물어볼 수 있어요!</p>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    # 버튼 자체에 스타일을 주기 위해 Streamlit 버튼 타입을 활용
    btn_types = ["primary", "secondary", "primary"]
    
    for i, q in enumerate(st.session_state.initial_questions):
        with cols[i]:
            if st.button(q, key=f"btn_{i}", type=btn_types[i % 3], use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# 대화 기록 출력
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and not msg["content"].startswith("### 📝 복습 퀴즈 타임!"):
            if msg.get("contexts"):
                with st.expander("🔍 참고한 핵심 내용 보기"):
                    for c in msg["contexts"]:
                        key_sentence = c.get('key_sentence', c['doc'][:80])
                        st.markdown(f"**출처:** {c['source']}  \n**핵심 문장:** {key_sentence}")
                        if c.get("link"):
                            st.markdown(f"**링크:** [웹페이지 이동]({c['link']})")
                        st.markdown("---")
        
        # 마지막 메시지가 assistant 이고 현재 새로운 질문 입력이 없을 때만 출력
        if idx == len(st.session_state.messages) - 1 and msg["role"] == "assistant" and not has_new_input:
            # 다시 설명 듣기 버튼 렌더링 (후속 질문보다 먼저)
            render_feedback_buttons(idx)
            
            if "followup_questions" in st.session_state:
                render_followup_questions(st.session_state.followup_questions, idx)

# 사용자 입력 처리
user_input = st.chat_input("선생님께 질문해 보세요!", key="chat_input_val")

# 초기 버튼 클릭 시 pending_question 처리
if 'pending_question' in st.session_state:
    user_input = st.session_state.pop('pending_question')

if user_input:
    # 1. 사용자 메시지 화면에 출력
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.query_count += 1
    
    # 2. 3단계 하이브리드 검색 실행
    contexts, search_stage = search_hybrid(user_input)
    
    # 3. 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("선생님이 생각 중이에요..."):
            system_prompt = get_system_prompt(section, search_stage)
            
            if model:
                # rag_pipeline.py 로직 참조하여 프롬프트 구성
                if contexts:
                    context_str = "\n\n".join([f"Source: {c['source']}\nContent: {c['doc']}" for c in contexts])
                    base_prompt = f"다음 지식을 바탕으로 학생의 질문에 답해주세요.\n\n[지식]\n{context_str}\n\n[학생의 질문]\n{user_input}"
                else:
                    base_prompt = f"[학생의 질문]\n{user_input}"
                
                full_prompt = f"{system_prompt}\n\n{base_prompt}"
                
                try:
                    response = model.generate_content(full_prompt)
                    bot_answer = response.text
                    # 답변이 생성된 직후에 각 Context에서 답변에 대응하는 핵심 문장을 추출하여 저장
                    if contexts:
                        for c in contexts:
                            c['key_sentence'] = extract_key_sentence(c['doc'], bot_answer)
                    elif search_stage == "Gemini":
                        import re
                        # 정규식을 유연하게 작성 (줄바꿈 허용, 공백 허용)
                        match = re.search(r'<source>\s*([\s\S]*?)\s*\|\s*([\s\S]*?)\s*</source>', bot_answer, re.IGNORECASE)
                        if match:
                            gemini_source = match.group(1).strip()
                            link = match.group(2).strip()
                            bot_answer = bot_answer.replace(match.group(0), "").strip()
                            contexts = [{"doc": "선생님이 가진 배경지식을 활용하여 작성한 답변입니다.", "source": f"제미나이 자체 지식 - {gemini_source}", "link": link, "key_sentence": "별도의 외부 문서 검색 없이 선생님의 지식으로 답변을 구성했습니다."}]
                        else:
                            # 만약 <source> 태그 형식이 아니지만 출처가 있을 수 있으므로 방어적 로직
                            bot_answer = re.sub(r'<source>[\s\S]*?</source>', '', bot_answer, flags=re.IGNORECASE).strip()
                            if "선생님은 사회 수업을 위한 챗봇이에요" not in bot_answer:
                                contexts = [{"doc": "선생님이 가진 배경지식을 활용하여 작성한 답변입니다.", "source": "제미나이 자체 답변", "link": "", "key_sentence": "별도의 외부 문서 검색 없이 선생님의 지식으로 답변을 구성했습니다."}]
                except Exception as e:
                    bot_answer = f"[오류] 답변 생성 중 문제가 발생했습니다: {e}"
            else:
                bot_answer = "[오류] AI 모델이 초기화되지 않았습니다."
            
            st.markdown(bot_answer)
            
            # 출처보기 Expander
            if contexts:
                with st.expander("🔍 참고한 핵심 내용 보기"):
                    for c in contexts:
                        key_sentence = c.get('key_sentence', c['doc'][:80])
                        st.markdown(f"**출처:** {c['source']}  \n**핵심 문장:** {key_sentence}")
                        if c.get("link"):
                            st.markdown(f"**링크:** [웹페이지 이동]({c['link']})")
                        st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_answer,
                "contexts": contexts
            })
            
            # 후속 질문 생성 및 저장
            fq = get_followup_questions(user_input, bot_answer)
            st.session_state.followup_questions = fq
            
            # 답변 재요청 버튼 렌더링 (후속 질문보다 먼저)
            render_feedback_buttons(len(st.session_state.messages) - 1)
            
            render_followup_questions(fq, len(st.session_state.messages) - 1)
            
            # 4. 구글 시트 로깅 (비동기)
            log_source = search_stage if search_stage == "Gemini" else contexts[0]["source"] if contexts else "Unknown"
            log_to_sheet_async(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                student_id=st.session_state.student_id,
                student_name=st.session_state.student_name,
                user_query=user_input,
                bot_response=bot_answer,
                source=log_source
            )
            
    # 5. 동적 퀴즈 생성 로직 (3회 질문마다)
    if st.session_state.query_count > 0 and st.session_state.query_count % 3 == 0:
        with st.chat_message("assistant"):
            st.markdown("### 📝 복습 퀴즈 타임!")
            with st.spinner("선생님이 퀴즈를 만들고 있어요..."):
                # 최근 6개 메시지 (질문3+답변3) 추출
                recent_logs = ""
                for m in st.session_state.messages[-6:]:
                    role = "학생" if m["role"] == "user" else "선생님"
                    recent_logs += f"{role}: {m['content']}\n"
                
                quiz_text = generate_quiz(recent_logs)
                if quiz_text:
                    if "[정답 및 해설]" in quiz_text:
                        parts = quiz_text.split("[정답 및 해설]")
                        q_and_options = parts[0]
                        answer_part = parts[1].strip()
                        
                        if "[보기]" in q_and_options:
                            q_parts = q_and_options.split("[보기]")
                            question_part = q_parts[0].replace("[문제]", "").strip()
                            options_part = q_parts[1].strip()
                            
                            st.markdown(question_part)
                            
                            # 보기는 별도의 색상 박스에 수직 배치
                            formatted_options = options_part.replace('\n', '<br>')
                            options_box = f"<div style='background-color: #E8F4F8; border: 1px solid #BFE0EC; padding: 15px; border-radius: 10px; margin: 15px 0; color: #1E6091; font-weight: 500; font-size: 15px; line-height: 1.6;'>{formatted_options}</div>"
                            st.markdown(options_box, unsafe_allow_html=True)
                            
                            with st.expander("✅ 정답 및 해설 확인하기"):
                                st.markdown(answer_part)
                                
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"### 📝 복습 퀴즈 타임!\n\n{question_part}\n\n{options_box}\n\n<details><summary>✅ 정답 및 해설 확인하기</summary>\n\n{answer_part}\n</details>"
                            })
                        else:
                            question_part = q_and_options.replace("[문제]", "").strip()
                            st.markdown(question_part)
                            
                            with st.expander("✅ 정답 및 해설 확인하기"):
                                st.markdown(answer_part)
                                
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"### 📝 복습 퀴즈 타임!\n\n{question_part}\n\n<details><summary>✅ 정답 및 해설 확인하기</summary>\n\n{answer_part}\n</details>"
                            })
                    else:
                        st.markdown(quiz_text)
                        st.session_state.messages.append({"role": "assistant", "content": f"### 📝 복습 퀴즈 타임!\n\n{quiz_text}"})
