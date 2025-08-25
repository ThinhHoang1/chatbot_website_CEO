import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # <-- Import thêm
import os
# Load .env
from dotenv import load_dotenv
load_dotenv()

# Lấy API key từ biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
# --- CẤU HÌNH ---
WEBSITE_URL = "https://www.ceo.pro.vn/" 

# --- HÀM TẢI DỮ LIỆU VÀ XỬ LÝ ---
@st.cache_resource
def load_and_process_data(url, api_key):
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
Hãy tuân thủ các quy tắc sau:
1. Chỉ sử dụng thông tin trong phần "Ngữ cảnh" dưới đây để trả lời.
2. Nếu thông tin không có trong ngữ cảnh, hãy trả lời một cách lịch sự rằng: "Tôi không tìm thấy thông tin này trên trang web." Đừng cố bịa ra câu trả lời.
3. Trả lời bằng tiếng Việt, một cách rõ ràng và chuyên nghiệp.

Ngữ cảnh:
{context}

Câu hỏi của người dùng: {question}

Câu trả lời của bạn:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Chatbot Website CEO Pro Club", page_icon="🤖")
st.title("🤖 Chatbot Hỗ Trợ Thông Tin CEO Pro Club")
st.caption(f"Tôi là trợ lý ảo, sẵn sàng trả lời các câu hỏi từ trang: {WEBSITE_URL}")

try:
    retriever = load_and_process_data(WEBSITE_URL, GOOGLE_API_KEY)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True # Trả về cả nguồn
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi là trợ lý ảo của trang web này. Bạn cần tìm thông tin gì ạ?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Bạn muốn hỏi gì về nội dung trang web?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
                response = qa_chain(prompt) # Gọi chain
                answer = response['result'] # Lấy câu trả lời
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
    st.info("Vui lòng kiểm tra lại Google API Key và URL của website.")