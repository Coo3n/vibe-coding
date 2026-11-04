import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

ENV_FILE = Path(__file__).with_name(".env")
REQUEST_TIMEOUT = 60

load_dotenv(ENV_FILE)

API_KEY = os.getenv("WB_API_KEY", "").strip()
API_URL = os.getenv("WB_API_URL", "").strip()
MODEL = os.getenv("WB_MODEL", "").strip()


def main() -> int:
    settings = {"WB_API_KEY": API_KEY, "WB_API_URL": API_URL, "WB_MODEL": MODEL}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        print(f"Заполните в .env: {', '.join(missing)}", file=sys.stderr)
        return 1

    prompt = input("Ваш запрос: ")
    request = Request(
        API_URL,
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            answer = json.load(response)
        print(answer["choices"][0]["message"]["content"])
    except HTTPError as error:
        print(f"Ошибка API: HTTP {error.code}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError):
        print("Не удалось подключиться к API. Проверьте URL и сеть.", file=sys.stderr)
        return 1
    except (ValueError, KeyError, IndexError, TypeError):
        print("Неожиданный формат ответа API.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
