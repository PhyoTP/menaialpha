from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("HCAI_API_KEY")

loader = PyMuPDFLoader("Unity Workshop June Slides.pdf")
documents = loader.load()

embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    base_url="https://ai.hackclub.com/proxy/v1",
    api_key=api_key,
)

vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever(
	 search_type="similarity",
	 search_kwargs={"k": 5}
	)

llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2",
    base_url="https://ai.hackclub.com/proxy/v1",
    api_key=api_key,  
)
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and latest user question, reformulate the question to be standalone. Do NOT answer it, just reformulate."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are MenAI, a coding companion for a Unity workshop.
You help students who are ahead or behind in the course.
Never give full code solutions - guide them to the answer instead.
Use the following context to answer the question.

{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

def respond(query, history):
    # query = input("Ask a question about the Unity workshop: ")
    response = retrieval_chain.invoke({"input": query, "chat_history": history})
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=response["answer"]))
    return response

if __name__ == "__main__":
    chat_history = []
    while True:
        query = input("Ask a question about the Unity workshop: ")
        print(respond(query, chat_history)["answer"])
