import gradio as gr
import os
import requests
import json

def generate_chat_response(user_message):
    url = "https://ki-toolbox.tu-braunschweig.de/api/v1/chat/send"
    api_key = os.getenv("API_KEY")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": user_message,
        "model": "gpt-4o"
    }

    # Sending the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # If the request was successful, return the reply from the model

        final_response = {}
        for line in response.iter_lines(decode_unicode=True):
            chunk = json.loads(line)
            if chunk.get("type") == "done":
                final_response = chunk
                break

        return final_response.get("response", "")
    else:
        # If there was an error, return the status code and error message
        return f"Error: {response.status_code}, {response.text}"

demo = gr.Interface(
    fn=generate_chat_response,
    inputs=gr.Textbox(label="Prompt", lines=10),
    outputs=gr.Textbox(label="Antwort", lines=30),
    title="Chatbot",
    description="Hier können Sie den Chatbot eine Frage stellen:",
)

demo.launch()