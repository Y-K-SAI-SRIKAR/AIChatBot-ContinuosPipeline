from pathlib import Path
import sys

# Ensure repository root is on sys.path for common imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from google.adk import Agent
from common.config_resolver import resolve_active_prompt_config

# Resolve active versioned instruction configuration safely
_resolved_prompt = resolve_active_prompt_config()
active_prompt_version = _resolved_prompt.prompt_version
instruction_text = _resolved_prompt.instruction_text

root_agent = Agent(
    name="ChatBot",
    description="AI ChatBot Agent",
    instruction=instruction_text,
    model="gemini-3-flash-preview",
    tools=[]
)
