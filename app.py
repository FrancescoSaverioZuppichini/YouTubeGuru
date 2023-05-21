import json
import logging
import os
from pathlib import Path
from typing import List
from uuid import uuid4

import gradio as gr
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from youtube_dl import YoutubeDL

os.environ["OPENAI_API_KEY"] = "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"
MODELS_NAMES = ["gpt-3.5-turbo", "gpt-4"]

logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s", level=logging.DEBUG
)


system_message = SystemMessage(content=Path("prompts/system.prompt").read_text())
human_message_prompt_template = HumanMessagePromptTemplate.from_template(
    Path("prompts/template.prompt").read_text()
)


def download_video_as_mp3(video_url: str, output_filename: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_filename,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def get_transcription(youtube_url: str):
    logging.info(f"Transcribing {youtube_url}")
    output_filename = Path(f"{str(uuid4())}.mp3")
    download_video_as_mp3(youtube_url, str(output_filename))
    logging.debug(f"video downloaded at {str(output_filename)}")
    with output_filename.open("rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    logging.info(f"Done!")
    output_filename.unlink()
    return transcript


def get_youtube_video_info(youtube_transcription: str, messages: List, chat):
    logging.info("Running GPT")
    human_message = human_message_prompt_template.format(
        youtube_transcription=youtube_transcription
    )
    messages.append(human_message)
    reply = chat(messages)
    messages.append(reply)
    logging.info(f"Done!")
    # we don't want the first ever message, too long
    chatbot_messages = [("", reply.content)]
    return chatbot_messages, messages


def run_message_on_chatbot(chat, message: str, chatbot_messages, messages):
    logging.info("asking question to GPT")
    messages.append(HumanMessage(content=message))
    reply = chat(messages)
    messages.append(reply)
    logging.debug(f"reply = {reply.content}")
    logging.info(f"Done!")
    chatbot_messages.append((message, messages[-1].content))
    return "", chatbot_messages, messages


def youtube_guru_button_handler(
    youtube_url: str, messages: List, temperature: float, model_name: str
):
    chat = ChatOpenAI(model_name=model_name, temperature=temperature)
    transcription = get_transcription(youtube_url)
    chatbot_messages, messages = get_youtube_video_info(transcription, messages, chat)
    return chatbot_messages, messages, chat


def on_clear_button_click():
    return "", [], [messages]


with gr.Blocks() as demo:
    messages = gr.State([system_message])
    youtube_transcription = gr.State("")
    model_selected = gr.State()
    chat = gr.State()

    with gr.Column():
        gr.Markdown("# Welcome to YouTubeGuru!")

        youtube_url = gr.Textbox(
            label="video url", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="chat input")
        msg.submit(
            run_message_on_chatbot,
            [chat, msg, chatbot, messages],
            [msg, chatbot, messages],
        )
        with gr.Row():
            with gr.Column():
                clear = gr.Button("Clear")
                clear.click(
                    on_clear_button_click,
                    [],
                    [youtube_transcription, chatbot, messages],
                    queue=False,
                )
            with gr.Accordion("Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperate",
                    interactive=True,
                )
                model_name = gr.Dropdown(
                    choices=MODELS_NAMES, value=MODELS_NAMES[0], label="model"
                )

        button = gr.Button("Run 🚀")
        button.click(
            youtube_guru_button_handler,
            inputs=[youtube_url, messages, temperature, model_name],
            outputs=[chatbot, messages, chat],
        )
