import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.schema import Document
from notion_client import Client
from notion_client.helpers import collect_paginated_api
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ✅ API 키
openai_api_key = st.secrets["OPENAI_API_KEY"]
notion = Client(auth=st.secrets["NOTION_TOKEN"])
notion_page_id = st.secrets["NOTION_PAGE_ID"]
SERVICE_ACCOUNT_FILE = "hrchatbot-459000-5ab15dc81cb8.json"  # 여긴 파일명이니까 그대로
SPREADSHEET_ID = st.secrets["GOOGLE_SHEET_ID"]
RANGE_NAME = st.secrets["SHEET_RANGE"]

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

# ✅ Google Sheets 텍스트 수집
def get_sheet_data():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        service = build("sheets", "v4", credentials=credentials)
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME
        ).execute()
        values = result.get("values", [])
        return "\n".join(["\t".join(row) for row in values]) if values else ""
    except Exception as e:
        print("⚠️ Sheets 오류:", e)
        return ""

# ✅ 데이터 수집 및 임베딩
documents = []
notion_text = get_notion_text(notion_page_id)
sheet_text = get_sheet_data()

if notion_text:
    documents.append(Document(page_content=notion_text))
if sheet_text:
    documents.append(Document(page_content=sheet_text))

if not documents:
    st.error("❗ 데이터 없음: Notion 또는 Google Sheets 연결 확인")
    st.stop()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), retriever=retriever)

# ✅ Streamlit UI
st.set_page_config(page_title="스파크플러스 HR 챗봇", layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])

# ✅ FAQ 버튼
with col3:
    st.markdown("### 📌 자주 묻는 질문")
    faq_list = [
        "본인 결혼 시 경조휴가는 며칠인가요?",
        "본인 결혼 시 경조지원금은 얼마인가요?",
        "배우자 출산 시 경조휴가는?",
        "부모 사망 시 경조휴가 일수는?"
    ]
    for q in faq_list:
        if st.button(q):
            st.session_state.faq_trigger = q
            st.rerun()

# ✅ 챗봇 메인
with col2:
    st.title("😀 스파크플러스 HR 챗봇")
    st.markdown("<p style='font-size:16px; color: gray;'>스파크플러스 인사제도 또는 Employee Services 관련 궁금한 점을 물어보세요 👇</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faq_trigger" not in st.session_state:
        st.session_state.faq_trigger = None

    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.faq_trigger = None
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("궁금한 점을 입력하세요")
    if query or st.session_state.faq_trigger:
        q = query if query else st.session_state.faq_trigger
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("user"):
            st.markdown(q)

        answer = qa.run(q)
        if not answer or answer.strip().lower() in ["i don't know.", "i do not know"]:
            answer = "저는 이 질문에 대한 정보가 없어서 정확한 답변을 드릴 수 없습니다."

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.session_state.faq_trigger = None
        st.rerun()