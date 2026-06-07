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
            "title":str("Sources: " + ", ".join([clean_context(i) for i in resp["context"]]))
        }
    })
    return "", gradio_history
def clean_context(doc):
    if doc.metadata.get("producer") == "PyPDF":
        return doc.metadata.get("title") + " page " + str(doc.metadata.get("page"))
    else:
        return doc.metadata.get("title")
with gr.Blocks() as demo:
    gr.Markdown("""# Menai
    For the BuildingBloCS June Jam Unity Workshop""")
    chatbot = gr.Chatbot(label="Menai", height="70vh")
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")

    msg.submit(respond_and_update, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()