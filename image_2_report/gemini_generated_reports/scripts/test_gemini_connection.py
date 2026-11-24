import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "gemini_generated_reports" / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

if not API_KEY:
    raise ValueError("Gemini API key is missing.")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=60000)
)

try:
    for attempt in range(1, 5):

        print(f"\nAttempt {attempt}/4: Testing {MODEL}")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents="Reply with exactly: Gemini API connection successful.",
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=50
                )
            )

            print("\nSUCCESS!")
            print(response.text)
            break

        except Exception as error:

            print(f"Error: {type(error).__name__}: {error}")

            if attempt == 4:
                print("\nAll attempts failed. Try again later.")
                break

            wait_time = 10 * (2 ** (attempt - 1))
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

finally:
    client.close()


import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "gemini_generated_reports" / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY or API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    raise ValueError("Gemini API key is missing from .env")

print(f"Testing Gemini API with model: {MODEL}", flush=True)
print("Connecting to Google...", flush=True)

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=30000
    )
)

try:
    response = client.models.generate_content(
        model=MODEL,
        contents="Reply with exactly: Gemini API connection successful.",
        config=types.GenerateContentConfig(
            max_output_tokens=50,
            temperature=0
        )
    )

    print("\nSUCCESS!")
    print(response.text)

except Exception as error:
    print("\nGemini API request failed:")
    print(type(error).__name__, str(error))

finally:
    client.close()