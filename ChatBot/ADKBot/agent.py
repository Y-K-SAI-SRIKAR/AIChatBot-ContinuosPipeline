from google.adk import Agent
from pathlib import Path


instruction_path = Path(__file__).parent / "instruction.txt"

with open(instruction_path, "r", encoding="utf-8") as f:
    instruction_text = f.read()

root_agent = Agent(
    name= "ChatBot",
    description="AI ChatBot Agent",
    instruction=instruction_text,
    model="gemini-3-flash-preview",
    tools=[]
)