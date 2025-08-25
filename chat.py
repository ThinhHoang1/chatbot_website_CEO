__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import asyncio
import streamlit as st
# Vẫn import các thành phần từ langchain-core hoặc langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Thay đổi các import để trỏ đến các gói cụ thể hơn
from langchain_community.document_loaders import WebBaseLoader # <-- THAY ĐỔI
from langchain_community.vectorstores import Chroma           # <-- THAY ĐỔI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
load_dotenv()

def sanitize_for_markdown(text: str) -> str:
    """
    Hàm này thay thế các ký tự có thể gây lỗi cho bộ render Markdown JavaScript
    trên một số trình duyệt di động. Cụ thể là dấu `?` trong các nhóm regex.
    """
    # Thay thế các ký tự có thể gây ra lỗi regex không hợp lệ
    # Ví dụ: một chuỗi như `(?<` có thể gây lỗi trên các trình duyệt cũ.
    # Ta có thể thay thế các ký tự đặc biệt hơn, nhưng bắt đầu với dấu `\` là phổ biến nhất.
    return text.replace('\\', '\\\\')

GOOGLE_API_KEY = st.secrets["GOOGLE_GEMINI_API_KEY"]
WEBSITE_URL = "https://www.ceo.pro.vn/"

# --- HÀM TẢI DỮ LIỆU VÀ XỬ LÝ ---
@st.cache_resource
def load_and_process_data(url, api_key):
    # TẠO VÀ SET EVENT LOOP MỚI CHO LUỒNG HIỆN TẠI
    asyncio.set_event_loop(asyncio.new_event_loop()) # <-- THÊM DÒNG NÀY

    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    vector_store = Chroma.from_documents(
        chunks,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

# --- TẠO PROMPT TEMPLATE ĐỂ SET ROLE ---
prompt_template = """Bạn là một trợ lý AI hữu ích và thân thiện, có nhiệm vụ trả lời các câu hỏi của người dùng chỉ dựa trên thông tin từ trang web được cung cấp.
Trả lời bằng tiếng Việt, một cách rõ ràng và chuyên nghiệp.

Ngữ cảnh:
{context}

Câu hỏi của người dùng: {question}

Câu trả lời của bạn:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Chatbot Website với Gemini", page_icon="🤖")
st.title("🤖 Chatbot Hỗ Trợ Thông Tin Website")
st.caption(f"Tôi là trợ lý ảo, sẵn sàng trả lời các câu hỏi từ trang: {WEBSITE_URL}")

try:
    retriever = load_and_process_data(WEBSITE_URL, GOOGLE_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi là trợ lý ảo của trang web này. Bạn cần tìm thông tin gì ạ?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Làm sạch tất cả nội dung trước khi render
            st.markdown(sanitize_for_markdown(message["content"]))


    if prompt := st.chat_input("Bạn muốn hỏi gì về nội dung trang web?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
                response_dict = qa_chain.invoke(prompt)
                raw_answer = response_dict['result']
                
                # Làm sạch câu trả lời trước khi hiển thị
                sanitized_answer = sanitize_for_markdown(raw_answer)
                
                st.markdown(sanitized_answer) # <-- Hiển thị nội dung đã được làm sạch
        
        # Lưu câu trả lời gốc (chưa làm sạch) vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": raw_answer})

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
    st.info("Vui lòng kiểm tra lại Google API Key và URL của website.")
