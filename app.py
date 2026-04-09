from model import respond
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage


chat_history = []
def chat(query, gradio_history):
    history = []
    for message in gradio_history:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
    response = respond(query, history)
    return f"""
    {response["answer"]}
    Sources:
    {"\n".join([i.page_content for i in response["context"]])}
    """
gr.ChatInterface(fn=chat).launch()