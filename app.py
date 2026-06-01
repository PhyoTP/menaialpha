from model import respond
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

def chat(query, gradio_history):
    history = []
    for message in gradio_history:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))

    return respond(query, history)

def respond_and_update(query, gradio_history):
    resp = chat(query, gradio_history)
    gradio_history.append({"role": "user", "content": query})
    gradio_history.append({
        "role": "assistant",
        "content": resp["answer"],
        "metadata": {
            "title":str(resp["context"])
        }
    })
    return "", gradio_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Menai", height="70vh")
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")

    msg.submit(respond_and_update, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()