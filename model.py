print("Starting")
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
print("Loaded dependencies")
load_dotenv()
api_key = os.getenv("HCAI_API_KEY")
pdf_name = "Unity Workshop June Slides-1"
pdf_loader = PyPDFLoader(pdf_name+".pdf")
pdf = pdf_loader.load()
print("Loaded slides")
url_loader = RecursiveUrlLoader(
    url="https://docs.unity3d.com/Manual/",
    max_depth=2,  # controls how deep it crawls
)
url = url_loader.load()
print("Loaded documentation")
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://ai.hackclub.com/proxy/v1",
    api_key=api_key,
)

if os.path.exists(pdf_name+"_index"):
    pdf_vector_store = FAISS.load_local(pdf_name+"_index", embeddings, allow_dangerous_deserialization=True)
else:
    pdf_vector_store = FAISS.from_documents(pdf, embeddings)
    pdf_vector_store.save_local(pdf_name+"_index")
url_vector_store = FAISS.from_documents(url, embeddings)
print("Created vector stores")
pdf_retriever = pdf_vector_store.as_retriever(
	search_type="similarity",
	search_kwargs={"k": 3}
)
url_retriever = url_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[pdf_retriever, url_retriever],
    weights=[0.7, 0.3]
)
llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2",
    base_url="https://ai.hackclub.com/proxy/v1",
    api_key=api_key,
)

# compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
# compressed_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=ensemble_retriever
# )
print("Created retrievers")
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and latest user question, reformulate the question to be standalone. Do NOT answer it, just reformulate."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, ensemble_retriever, contextualize_prompt
)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Menai, a coding companion for a Unity workshop.
You help students who are ahead or behind in the course.
Never give full code solutions - guide them to the answer instead.

You have access to two knowledge sources:
- Workshop slides (curriculum): use this to understand what the student should know at their current lesson, pace your explanations accordingly, and avoid jumping ahead
- Unity documentation: use this for accurate technical details, API references, and code explanations

{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
print("Model loaded")
def respond(query, history):
    # query = input("Ask a question about the Unity workshop: ")
    response = retrieval_chain.invoke({"input": query, "chat_history": history})
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=response["answer"]))
    return response

if __name__ == "__main__":
    chat_history = []
    while True:
        user_query = input("Ask a question about the Unity workshop: ")
        print(respond(user_query, chat_history)["answer"])
