from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import StructuredTool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Google Search Tool
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
search = GoogleSearchAPIWrapper(google_api_key=google_api_key,google_cse_id=google_cse_id)

def search_function(query: str):
    print("Searching for:", query)
    print("\n")
    return search.run(query)

gsearch = StructuredTool.from_function(
    func=search_function,
    name="Google Search",
    description="Search Google for the price of a laptop on indian marletplaces like Amazon, Flipkart, Croma, Reliance Digital etc.",
)

# GROQ Mixtral LLM
model = "mixtral-8x7b-32768"
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

# Retreiver Tool
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2", model_kwargs={'device':'cpu'})
db=FAISS.load_local("vectorstore/db_faiss",embeddings,allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={"k": 7})
retreiver_tool=create_retriever_tool(retriever,"laptopdata_tool","Retrieve any information about specs of a laptop")

#Agent and Tools
tools=[gsearch,retreiver_tool]
prompt=qa_prompt = hub.pull("trial")
if 'memory' not in st.session_state:
    st.session_state['memory'] = ChatMessageHistory(session_id="test-session")
if 'config' not in st.session_state:
    st.session_state['config'] = {"configurable": {"session_id": "test-session"}}
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: st.session_state['memory'],
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Follwing code snippet is for testing the agent in the terminal
# user_prompt=input("Enter your query:")
# while(prompt!="exit"):
#     print(agent_with_chat_history.invoke({"input": user_prompt},st.session_state['config'])["output"])
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
            ai_response = agent_with_chat_history.invoke({"input": user_prompt},st.session_state['config'])["output"]
            print(ai_response)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
