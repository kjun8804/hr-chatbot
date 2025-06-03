import streamlit as st
import os
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from notion_client import Client
from notion_client.helpers import collect_paginated_api
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.vectorstores import DocArrayInMemorySearch

# ✅ API 키
openai_api_key = st.secrets["OPENAI_API_KEY"]
notion = Client(auth=st.secrets["NOTION_TOKEN"])
notion_page_id = st.secrets["NOTION_PAGE_ID"]
SPREADSHEET_ID = st.secrets["GOOGLE_SHEET_ID"]

# ✅ 질문 카테고리 키워드
category_keywords = {
    "근무일반": [
        "출근", "퇴근", "근무시간", "유연근무", "지각", "결근", "조퇴",
        "재택", "외근", "근무복장", "근무제도", "시차출근", "연장근무",
        "사내규정", "일반규정", "매뉴얼", "직무규칙", "서약서"
    ],
    "급여보상": [
        "급여", "월급", "임금", "연봉", "보너스", "성과급", "상여",
        "인센티브", "수당", "퇴직금", "연말정산", "급여일", "세금",
        "지급일", "급여명세서", "세율", "소득", "보상제도"
    ],
    "복무": [
        "연차", "반차", "병가", "공가", "휴직", "무급휴가", "특별휴가",
        "대체휴가", "생리휴가", "결근", "유급", "지각", "조퇴",
        "복무", "출결처리", "근태", "출석", "업무복귀", "휴가사용", "생일", 
    ],
    "복지제도": [
        "복지", "건강검진", "식대", "명절선물", "경조사비", "동호회", "장기근속",
        "자기계발", "사내복지", "리프레시", "워라밸", "법인휴양소", "배우자","배우자출산","경조지원금",
        "기프티콘", "간식", "헬스", "생활비지원", "생일선물", "슾머니", "경조휴가", "출산휴가"
    ],
    "업무지원경비": [
        "장비", "노트북", "데스크탑", "소프트웨어", "계정", "메일",
        "설치", "반납", "VPN", "구입", "청구", "법인카드", "비용정산",
        "업무비", "출장비", "증빙", "구매요청", "영수증"
    ],
    "채용온보딩": [
        "채용", "입사", "합격", "면접", "전형", "이력서", "경력", "신입",
        "온보딩", "입사일", "채용절차", "채용서류", "복수합격",
        "입사 전 안내", "근로계약", "계정생성", "사번", "출근일"
    ]
}

def classify_question(question):
    for category, keywords in category_keywords.items():
        if any(keyword in question for keyword in keywords):
            return category
    return "근무일반"

# ✅ Notion 텍스트 수집
def get_notion_text(page_id):
    try:
        children = collect_paginated_api(notion.blocks.children.list, block_id=page_id)
        texts = []
        for child in children:
            rich = child.get(child["type"], {}).get("rich_text", [])
            for part in rich:
                texts.append(part.get("plain_text", ""))
        return "\n".join(texts)
    except Exception as e:
        print("⚠️ Notion 오류:", e)
        return ""

# ✅ 특정 시트에서만 불러오기 (카테고리별)
from langchain.schema import Document

# ✅ get_sheet_data_by_category: try ~ except 구조 완성
def get_sheet_data_by_category(category):
    try:
        sheet_range = f"'{category}'!A1:Z1000" if "." in category else f"{category}!A1:Z1000"
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(st.secrets["GOOGLE_CREDENTIALS"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        service = build("sheets", "v4", credentials=credentials)
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=sheet_range
        ).execute()
        values = result.get("values", [])

        if values:
           return [
                Document(
                    page_content=row[1],
                    metadata={"source": category, "question": row[0]}
                )
                for row in values if len(row) > 1
            ]

        else:
            return []

    except Exception as e:
        print(f"⚠️ Sheets 오류({category}):", e)
        return []

from difflib import get_close_matches

from difflib import SequenceMatcher

def find_fuzzy_match(user_q, sheet_docs):
    best_match = None
    highest_ratio = 0.0

    for doc in sheet_docs:
        question = doc.metadata.get("question", "")
        ratio = SequenceMatcher(None, user_q.strip().lower(), question.strip().lower()).ratio()

        if ratio == 1.0:
            # ✅ 완전일치 → 즉시 반환
            return [doc]
        elif ratio >= 0.8 and ratio > highest_ratio:
            # ✅ 90% 이상 유사 → 가장 유사한 항목 저장
            best_match = doc
            highest_ratio = ratio

    if best_match:
        return [best_match]

    return []

def contains_all_keywords(original: str, rewritten: str) -> bool:
    original_tokens = original.lower().split()
    return all(token in rewritten.lower() for token in original_tokens)

# ✅ 데이터 수집 및 임베딩
documents = []
notion_text = get_notion_text(notion_page_id)

if notion_text:
    documents.append(Document(page_content=notion_text))

# ✅ (이제는 get_sheet_data()를 호출하지 않음!)

if not documents:
    st.error("❗ 데이터 없음: Notion 또는 Google Sheets 연결 확인")
    st.stop()

# ✅ 문서 쪼개기
splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = splitter.split_documents(documents)

# ✅ 벡터 저장소 및 임베딩
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# ✅ filter 정의도 필요:
filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=filter,
    base_retriever=retriever,
)

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain

# ✅ 최신 ChatOpenAI 사용
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4-1106-preview",
    temperature=0.3,
    max_tokens=1024
)

# ✅ rewrite_chain 위치는 여기!!
rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""당신은 HR 제도에 대해 안내하는 챗봇입니다. 사용자의 질문을 회사 내부 매뉴얼에서 검색하기 적합한 형태로 다시 써주세요. 너무 짧거나 모호한 질문이라도 인사팀에 물을 만한 표현으로 명확하게 바꾸세요.

따라서 사용자의 의도를 바탕으로, 회사의 인사 매뉴얼, 복무 규정, 복지제도 등에서 관련 내용을 찾기 적합한 형태로 질문을 다시 작성해주세요.

- 가능한 한 명확하고 구체적으로 바꿔주세요.
- 질문자가 물어보는 질문 중 단어 또는 고유명사들의 경우 별도 가공하지 않고 검색하는 주요한 키워드로 사용해주세요. 예시 - 슾머니, 배우자출산휴가 등
- 질문자가 물어본 질문에 대해 노출할 떄는 그대로 노출되게 해주세요. 절대 질문자의 질문을 마음대로 가공해서 노출시키지마세요.
- 답변의 소스의 경우, 외부에서 절대로 가져오지 말고 제공된 노션과 스프레드시트에 있는 것만 답변해주세요. 
- 문서에 없는 내용을 추측하거나 임의로 생성하지 마세요.
- 사용자의 질문에 등장한 단어는 특히 고유명사(예: 슾머니, 부서명, 제도명 등)는 절대 바꾸지 말고, **철자, 띄어쓰기, 형태 그대로 유지**하세요.
- 의미를 유추하거나 일반화해서 변경하지 마세요. 특히, 사용자가 오타를 포함해 입력한 고유명사도 그대로 사용하세요.
- 예: "슾머니"는 "스폿머니"로 바꾸면 안 됩니다.
- 질문 문장은 전체적으로 부드럽게 바꾸되, **원문의 명칭(고유명사)은 반드시 그대로 유지**하세요.

예시:
입력: 배우자 출산 휴가는 며칠?
→ 출력: 배우자 출산 시 사용할 수 있는 경조휴가는 며칠인가요?

입력: 휴가 종류
→ 출력: 회사에서 사용할 수 있는 휴가의 종류에는 무엇이 있나요?

입력: 생일 관련 복지 뭐 있어요?
→ 출력: 생일에 제공되는 복지 혜택에는 어떤 것들이 있나요?

입력: 본인 결혼 시 경조휴가는 며칠인가요?
→ 출력: 본인 결혼 시 제공되는 경조휴가 휴가일수는 며칠인가요?

---

사용자 질문: {question}
명확한 질문:
"""
)
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

# ✅ Prompt 구성
question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""다음은 회사의 내부 HR 정책 문서입니다. 이 문서에 기반해 사용자의 질문에 답변하세요.

❗ 중요한 규칙:
- 문서 내용만을 기반으로 답변해야 하며, 외부 지식이나 일반 상식은 절대 사용하지 마세요.
- 문서에 명확한 정보가 없는 경우, 반드시 "제공된 자료에 없습니다. HR팀에 문의해주세요."라고 답변하세요.
- 문서에 없는 내용을 추론하거나 유추하지 마세요.
- "회사 정책에 따라", "근로기준법상", "국가별로 다르다" 같은 문구도 문서에 없으면 절대 사용하지 마세요.

문서:
{context}

질문:
{question}

답변:"""
)

refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "context", "question"],
    template="""기존 답변을 아래 문서 내용을 반영해 더 나은 답변으로 수정해주세요.

❗ 주의: 문서에 없는 내용은 절대 보완하거나 생성하지 마세요.
❗ 문서에 근거 없는 내용은 제거하고, 문서에 포함된 내용만 사용하세요.

기존 답변: {existing_answer}

추가 문서: {context}

질문: {question}
수정된 답변:"""
)

# ✅ LLM 체인 구성
initial_llm_chain = LLMChain(llm=llm, prompt=question_prompt)
refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)

combine_documents_chain = RefineDocumentsChain(
    initial_llm_chain=initial_llm_chain,
    refine_llm_chain=refine_llm_chain,
    document_variable_name="context",
    initial_response_name="existing_answer"
)

# ✅ RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compressed_retriever,
    chain_type="stuff",  # ✅ stuff로 바꾸기
    return_source_documents=False,
    chain_type_kwargs={
        "prompt": question_prompt
    }
)

# ✅ Streamlit UI
st.set_page_config(page_title="스파크플러스 HR 챗봇", layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])

# ✅ col3: FAQ 버튼 클릭 시 질문 기록
with col3:
    st.markdown("### 📌 자주 묻는 질문")
    faq_list = [
        "본인 결혼 시 경조휴가는 며칠인가요?",
        "본인 결혼 시 경조지원금은 얼마인가요?",
        "배우자 출산 시 경조휴가는?",
        "부모 사망 시 경조휴가 일수는?",
        "슾머니 지급 기준은?"
    ]
    for q in faq_list:
        if st.button(q):
            st.session_state.messages.append({"role": "user", "content": q})
            rewritten_input = rewrite_chain.run(q)
            if not contains_all_keywords(q, rewritten_input):
                rewritten_input = q
            st.session_state.faq_trigger = rewritten_input
            st.rerun()

# ✅ col2 내부
with col2:
    st.title("😀 스파크플러스 HR 챗봇")
    st.markdown("<p style='font-size:16px; color: black;'>스파크플러스 인사제도 또는 Employee Services 관련     궁금한 점을 물어보세요 👇</p>", unsafe_allow_html=True)

    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.faq_trigger = None
        st.rerun()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faq_trigger" not in st.session_state:
        st.session_state.faq_trigger = None

    # 과거 메시지 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ✅ 사용자 입력창
    user_input = st.chat_input("궁금한 점을 입력하세요")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        rewritten_input = rewrite_chain.run(user_input)
        if not contains_all_keywords(user_input, rewritten_input):
            rewritten_input = user_input
        st.session_state.faq_trigger = rewritten_input
        st.rerun()


# ✅ col2 바깥: 질문 처리
q = None
if "faq_trigger" in st.session_state and st.session_state.faq_trigger:
    q = st.session_state.faq_trigger
    st.session_state.faq_trigger = None

if q:
    answer = ""
    selected_category = classify_question(q)
    sheet_documents = get_sheet_data_by_category(selected_category)

    # ✅ Step 0: 시트 질문 정확 일치 확인
    for doc in sheet_documents:
        if doc.metadata["question"].strip() == q.strip():
            answer = doc.page_content.strip()
            answer += "\n\n[🔗 제도상세: 바로가기](https://zrr.kr/jBwxck)"
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    if sheet_documents:
        # Step 1: 유사 질문 우선 매칭
        fuzzy_match = find_fuzzy_match(q, sheet_documents)
        if fuzzy_match:
            answer = fuzzy_match[0].page_content.strip()
        else:
            # Step 2: GPT rewrite 후 검색
            rewritten_q = rewrite_chain.run(q)
            selected_category = classify_question(rewritten_q)
            sheet_documents = get_sheet_data_by_category(selected_category)

            if not sheet_documents:
                answer = "❌ 관련 정보를 찾을 수 없습니다. HR팀에 문의해주세요."
            else:
                vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

                # ✅ filter 정의도 필요:
                filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

                compressed_retriever = ContextualCompressionRetriever(
                    base_compressor=filter,
                    base_retriever=retriever,
                )
                retrieved_docs = compressed_retriever.get_relevant_documents(rewritten_q)

                if not retrieved_docs:
                    answer = "❌ 관련 정보를 찾을 수 없습니다. HR팀에 문의해주세요."
                else:
                    raw_answer = qa.run(rewritten_q)

                    # ✅ 만약 답변에 "해당 내용은 제공된 자료에 없습니다"가 없다면, 무조건 sheet 첫 문단 사용
                    if (
                        not raw_answer or
                        raw_answer.strip().lower() in ["i don't know.", "i do not know"] or
                        "제공된 자료에 없습니다" not in raw_answer and len(raw_answer.strip()) < 50 or
                        "회사마다 다릅니다" in raw_answer or
                        "남녀고용평등법" in raw_answer or 
                        "근로기준법" in raw_answer or 
                        "노동법" in raw_answer or 
                        "회사 정책에 따라" in raw_answer  # ← GPT 외부 지식 기반 표현 필터링
                    ):
                        answer = retrieved_docs[0].page_content.strip()
                    elif raw_answer:
                        answer = raw_answer.strip()
                    else:
                        answer = "❌ 관련 정보를 찾을 수 없습니다. HR팀에 문의해주세요."

    # ✅ 링크 추가
    answer += "\n\n[🔗 제도상세: 바로가기](https://zrr.kr/jBwxck)"

    # ✅ 답변 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
