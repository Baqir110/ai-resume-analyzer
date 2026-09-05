import os
from dotenv import load_dotenv
from app.services.llm_provider import LLMService
from app.services.llm_provider import LLMService


def test_gateway_connection():
    assert LLMService is not None


load_dotenv()

print("Testing connection to Experiential Labs Gateway...")

try:
    # Test generation using the gateway via OpenAI-compatible provider
    response = LLMService.generate(
        prompt="Say 'Gateway connection successful!' and nothing else.",
        provider="openai",
    )
    print("\nResult from Gateway:")
    print(response)
except Exception as e:
    print(f"\nTest failed: {e}")
