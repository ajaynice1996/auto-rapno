import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_DIR = SCRIPT_DIR.parent
ENV_FILE = GEMINI_DIR / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=60000)
)

try:
    for attempt in range(1, 5):
        print(f"\nAttempt {attempt}/4: {MODEL}")

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents="Reply with exactly: Gemini API connection successful.",
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=100
                )
            )

            print("\nResponse received.")

            if response.text:
                print("SUCCESS!")
                print(response.text)
                break

            print("WARNING: Response contains no directly accessible text.")
            print("Candidate information:")

            if response.candidates:
                for candidate in response.candidates:
                    print("Finish reason:", candidate.finish_reason)
                    print("Content:", candidate.content)
            else:
                print("No candidates returned.")

            raise RuntimeError("Gemini returned no text.")

        except Exception as error:
            print(f"Error: {type(error).__name__}: {error}")

            if attempt == 4:
                raise

            wait_time = 10 * (2 ** (attempt - 1))
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

finally:
    client.close()
