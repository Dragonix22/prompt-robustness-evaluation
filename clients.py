import os
import re
import time

from google import genai


MODEL_NAME = "gemini-3.1-flash-lite"

REQUEST_INTERVAL = 5.0
MAX_RETRIES = 5


def get_api_key() -> str:

    key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if key:
        return key

    try:
        from google.colab import userdata

        key = userdata.get(
            "gai-studio-key"
        )

        if key:
            return key

    except Exception:
        pass

    raise RuntimeError(
        "Gemini API key not found. "
        "Set GEMINI_API_KEY or add "
        "'gai-studio-key' to Colab Secrets."
    )


client = genai.Client(
    api_key=get_api_key()
)


def parse_retry_seconds(error: Exception) -> float:

    text = str(error)

    match = re.search(
        r"retry in ([0-9.]+)s",
        text,
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1))

    return 30.0


def call_model(prompt: str) -> str:

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0,
                },
            )

            time.sleep(REQUEST_INTERVAL)

            return response.text.strip()

        except Exception as exc:

            if (
                "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
            ):

                wait = max(
                    REQUEST_INTERVAL,
                    parse_retry_seconds(exc),
                )

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait:.1f}s..."
                )

                time.sleep(wait)

                continue

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(
                REQUEST_INTERVAL * (attempt + 1)
            )

    raise RuntimeError(
        "Model request failed after retries."
    )
