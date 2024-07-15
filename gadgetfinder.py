from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
import google.generativeai as genai
import streamlit as st

llm=GoogleGenerativeAI(model="gemini-1.0-pro-latest")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2", model_kwargs={'device':'cpu'})
db=FAISS.load_local("vectorstore/db_faiss",embeddings,allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={"k": 7})

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = hub.pull("gadgetfinder")
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


if 'store' not in st.session_state:
    st.session_state['store'] = {}
    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:
        st.session_state['store'][session_id] = ChatMessageHistory()
    return st.session_state['store'][session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
# prompt=input("Enter your query:")
# while(prompt!="exit"):
#     print(conversational_rag_chain.invoke(
#         {"input": prompt},
#         config={
#             "configurable": {"session_id": "abc123"}
#         },  # constructs a key "abc123" in `store`.
#     )["answer"])
#     print("\n")
#     prompt=input()

st.set_page_config(
    page_title="Gadget Finder",
    page_icon="🤖",
    layout="wide"
)


st.title("Gadget Finder")


# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, Tell me about the purpose of buying a laptop and your budget in INR."}
    ]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = conversational_rag_chain.invoke({"input": user_prompt},config={"configurable": {"session_id": "abc123"}},)
            st.write(ai_response["answer"])
    new_ai_message = {"role": "assistant", "content": ai_response["answer"]}
    st.session_state.messages.append(new_ai_message)
