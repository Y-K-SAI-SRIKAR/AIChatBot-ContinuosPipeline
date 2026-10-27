from pathlib import Path
import os
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
loaded = load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from pydantic import BaseModel

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from ADKBot.agent import root_agent

app = FastAPI(title="AI ChatBot API")


class ChatRequest(BaseModel):
    message: str


session_service = InMemorySessionService()

runner = Runner(
    app_name="WPChatBot",
    agent=root_agent,
    session_service=session_service,
    auto_create_session=True
)

@app.get("/")
def home():
    return {
        "status": "AI ChatBot API Running"
    }

@app.post("/chat")
async def chat(request: ChatRequest):

    response_text = ""

    async for event in runner.run_async(
        user_id="web-user",
        session_id="default-session",
        new_message=types.UserContent(
            parts=[
                types.Part(text=request.message)
            ]
        )
    ):

        if event.content and event.content.parts:

            for part in event.content.parts:

                if getattr(part, "text", None):
                    response_text += part.text

    return {
        "reply": response_text
    }