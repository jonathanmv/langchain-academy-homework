from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file from root directory (one level up)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

msg = HumanMessage(content="Hello, world! I'm Jonathan")
system_msg = SystemMessage(content="Please reply to the user including your model name and version")

messages = [system_msg, msg]

print(gpt4o_chat.invoke(messages))
print(gpt35_chat.invoke(messages))
