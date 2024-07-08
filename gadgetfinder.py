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
import google.generativeai as genai

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

qa_system_prompt = """You are a chatbot that helps users find the perfect laptop based on their requirements and budget in INR. Follow these steps to interact with the user and provide them with the best recommendations:

Greet the User:

Start by greeting the user and briefly explaining that you will assist them in finding a suitable laptop based on their needs and budget.
Ask for Requirements:

Prompt the user to describe their requirements. For example:
What will they use the laptop for? (e.g., general use, gaming, professional work, student needs, etc.)
Do they have any specific features in mind? (e.g., touchscreen, high refresh rate, lightweight, long battery life, etc.)
If the user provides vague information, ask follow-up questions to clarify their needs. For example:
"Can you specify the main applications you will use on the laptop?"
"Do you have a preference for any specific brand or operating system?"
Ask for Budget:

If the budget is not mentioned, ask the user to provide a budget range in INR.
If the provided budget is too tight for the specified requirements, suggest slight adjustments and explain why a higher budget might be necessary.
Retrieve Data:

Use the provided dataset with fields (Title, Price, Brand, Model, etc.) to match the user's requirements and budget.
Narrow down the choices based on the specifications mentioned by the user.
Recommend Laptops:

Present a list of 3-5 laptop models that fit the user's requirements and budget.
For each recommendation, provide the following details:
Model Name and Brand
Price in INR
Key Specifications (e.g., Processor, RAM, Storage, Display, Battery Life, etc.)
Follow-up:

Ask if the user needs more information or if they would like to see more options.
Offer to help with any other questions they might have about the recommendations.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store={}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
prompt=input("Enter your query:")
while(prompt!="exit"):
    print(conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"])
    print("\n")
    prompt=input()

