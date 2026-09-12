"""Ad-hoc smoke test for the LLM gateway."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("EXPERIENTIAL_ORG_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

print("Target Base URL:", client.base_url)

response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "ping"}],
)

print("Response received successfully through gateway.")
