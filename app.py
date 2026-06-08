from model import respond
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from data_storage import save_message

def chat(query, gradio_history):
    history = []
    for message in gradio_history:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))

    return respond(query, history)

def respond_and_update(query, gradio_history, session_id):
    resp = chat(query, gradio_history)
    gradio_history.append({"role": "user", "content": query})
    gradio_history.append({
        "role": "assistant",
        "content": resp["answer"],
        "metadata": {
            "title":str("**Sources**: " + ", ".join([clean_context(i) for i in resp["context"]]))
        }
    })
    save_message(session_id, "user", query)
    save_message(session_id, "assistant", resp["answer"])
    return "", gradio_history
def clean_context(doc):
    if doc.metadata.get("producer") == "PyPDF":
        return doc.metadata.get("title") + " page " + str(doc.metadata.get("page"))
    else:
        return "["+doc.metadata.get("title")+"]("+doc.metadata.get("source")+")"
with gr.Blocks() as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))
    gr.Markdown("""# Menai
    For the BuildingBloCS June Jam Unity Workshop""")
    chatbot = gr.Chatbot(label="Menai", height="70vh")
    msg = gr.Textbox(placeholder="Press enter ⏎ to send", label="Your Message")
    msg.submit(respond_and_update, inputs=[msg, chatbot, session_id], outputs=[msg, chatbot])
    gr.Markdown("Menai is currently in its alpha stage, so bugs may occur!<br>Your chats will be stored to improve user experiences.")
demo.launch()