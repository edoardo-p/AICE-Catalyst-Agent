import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

load_dotenv()


endpoint = "https://catalyst-agent.openai.azure.com/"
deployment = "gpt-4o-mini"
api_version = "2025-01-01-preview"
api_key = os.environ.get("API_KEY")

chat = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_version=api_version,
    api_key=api_key,
)

# Example
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="I am going to Paris, what should I see?"),
]
resp = chat.invoke(messages)
print(resp.content)
