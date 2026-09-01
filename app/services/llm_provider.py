import os
import requests
from typing import Optional
from google import genai
import openai
import anthropic
from groq import Groq


class LLMService:
    @staticmethod
    def generate(
        prompt: str, 
        provider: str = "gemini", 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None
    ) -> str:
        provider = provider.lower()

        try:
            # 1. GOOGLE GEMINI
            if provider == "gemini":
                key = api_key or os.getenv("GEMINI_API_KEY")
                if not key:
                    raise ValueError("Gemini API key is required.")
                client = genai.Client(api_key=key)
                model = model_name or "gemini-2.5-flash"
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                return response.text

            # 2. GROQ
            elif provider == "groq":
                key = api_key or os.getenv("GROQ_API_KEY")
                if not key:
                    raise ValueError("Groq API key is required.")
                client = Groq(api_key=key)
                model = model_name or "openai/gpt-oss-120b"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            # 3. OPENROUTER
            elif provider == "openrouter":
                key = api_key or os.getenv("OPENROUTER_API_KEY")
                if not key:
                    raise ValueError("OpenRouter API key is required.")
                client = openai.OpenAI(
                    api_key=key, 
                    base_url="https://openrouter.ai/api/v1"
                )
                model = model_name or "deepseek/deepseek-chat"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            # 4. DEEPSEEK
            elif provider == "deepseek":
                key = api_key or os.getenv("DEEPSEEK_API_KEY")
                if not key:
                    raise ValueError("DeepSeek API key is required.")
                client = openai.OpenAI(api_key=key, base_url="https://api.deepseek.com")
                model = model_name or "deepseek-chat"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            # 5. OPENAI
            elif provider == "openai":
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError("OpenAI API key is required.")
                client = openai.OpenAI(api_key=key)
                model = model_name or "gpt-4o-mini"
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            # 6. ANTHROPIC
            elif provider == "claude":
                key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not key:
                    raise ValueError("Anthropic API key is required.")
                client = anthropic.Anthropic(api_key=key)
                model = model_name or "claude-3-5-haiku-20241022"
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            # 7. OLLAMA
            elif provider == "ollama":
                url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
                model = model_name or "llama3"
                payload = {"model": model, "prompt": prompt, "stream": False}
                res = requests.post(url, json=payload, timeout=90)
                if res.status_code == 200:
                    return res.json().get("response", "")
                raise RuntimeError(f"Ollama connection error: {res.text}")

            else:
                raise ValueError(f"Unsupported AI provider: {provider}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"AI Provider Error ({provider.upper()}): {str(e)}")