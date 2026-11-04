import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from dotenv import load_dotenv
from jsonschema import Draft202012Validator

load_dotenv(Path(__file__).with_name(".env"))

API_KEY = os.getenv("LLM_API_KEY", "").strip()
API_URL = os.getenv("LLM_API_URL", "").strip()
MODEL = os.getenv("LLM_MODEL", "").strip()
REQUEST_TIMEOUT = 60
RUNS = 3
MAX_TOKENS = 256
STOP = "<END>"
PROMPT = (
    "Дай три совета начинающему Python-разработчику. "
    "У каждого совета должны быть короткий заголовок и объяснение в одном предложении."
)
FORMAT = (
    'Ответь только JSON-объектом вида {"tips": [{"title": "...", "explanation": "..."}]}. '
    "Массив tips должен содержать ровно три совета. Без Markdown и текста вне JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "tips": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["title", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tips"],
    "additionalProperties": False,
}
MODES = ("baseline", "format", "max_tokens", "stop", "schema")
VALIDATOR = Draft202012Validator(SCHEMA)


def build_payload(mode, max_tokens):
    messages = []
    if mode != "baseline":
        instruction = FORMAT
        if mode == "stop":
            instruction += f" После завершённого JSON выведи маркер {STOP}."
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": PROMPT})
    payload = {"model": MODEL, "messages": messages}
    if mode in ("max_tokens", "stop", "schema"):
        payload["max_tokens"] = max_tokens
    if mode == "stop":
        payload["stop"] = [STOP]
    if mode == "schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "python_tips", "strict": True, "schema": SCHEMA},
        }
    return payload


def send_request(payload):
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


def inspect_answer(response):
    choice = response["choices"][0]
    message = choice["message"]
    if not isinstance(message, dict):
        raise ValueError("message должен быть объектом")
    content = message.get("content")
    result = {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "refusal": message.get("refusal"),
        "usage": response.get("usage"),
        "json_valid": False,
        "schema_valid": False,
        "validation_error": None,
    }
    if not isinstance(content, str) or not content.strip():
        result["validation_error"] = "Нет текстового ответа (пустой content или отказ)."
        return result
    try:
        parsed = json.loads(content)
    except ValueError as error:
        result["validation_error"] = f"JSON: {error}"
        return result
    result["json_valid"] = True
    result["parsed"] = parsed
    errors = list(VALIDATOR.iter_errors(parsed))
    result["schema_valid"] = not errors
    if errors:
        result["validation_error"] = "; ".join(error.message for error in errors)
    return result


def summarize(records):
    summary = {}
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        answers = [row for row in rows if "error" not in row]
        summary[mode] = {
            "attempts": len(rows),
            "api_errors": len(rows) - len(answers),
            "json_valid": sum(row["json_valid"] for row in answers),
            "schema_valid": sum(row["schema_valid"] for row in answers),
            "truncated": sum(row["finish_reason"] == "length" for row in answers),
            "refusals": sum(bool(row["refusal"]) for row in answers),
        }
    return summary


def conclusions(summary):
    lines = []
    for mode, counts in summary.items():
        attempts = counts["attempts"]
        if not attempts:
            continue
        answers = attempts - counts["api_errors"]
        lines.append(
            f"{mode}: попыток {attempts}, ошибок API {counts['api_errors']}, "
            f"JSON {counts['json_valid']}/{answers}, схема {counts['schema_valid']}/{answers}, "
            f"обрезано {counts['truncated']}, отказов {counts['refusals']}."
        )
        if answers > counts["json_valid"]:
            lines.append("  Есть ответы, которые json.loads не принимает, или пустой content.")
        if counts["json_valid"] > counts["schema_valid"]:
            lines.append("  Корректный JSON не гарантирует нужные поля, типы и число советов.")
        if counts["truncated"]:
            lines.append("  finish_reason=length: исчерпан лимит токенов; ответ может быть незавершён.")
        if mode == "schema" and counts["api_errors"]:
            lines.append("  Смотрите ошибки API: схема могла быть отвергнута, но это не доказано одним HTTP-кодом.")
    lines.append(
        "Вывод относится только к этим прогонам. Успешный JSON не доказывает поддержку "
        "strict на стороне сервера; схема не гарантирует смысловую правильность. "
        "finish_reason=stop не отличает stop sequence от естественного завершения."
    )
    return lines


def save_report(path, report):
    report["summary"] = summarize(report["records"])
    report["conclusions"] = conclusions(report["summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(json.dumps(report, ensure_ascii=False, indent=2))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Сравнение уровней контроля ответа LLM")
    parser.add_argument("--runs", type=int, default=RUNS, help="Прогоны каждого режима (по умолчанию 3)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--output", type=Path, help="Файл JSON с результатами")
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("--runs и --max-tokens должны быть положительными")
    settings = {"LLM_API_KEY": API_KEY, "LLM_API_URL": API_URL, "LLM_MODEL": MODEL}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        print(f"Заполните в .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("results") / f"{stamp}.json"
    # An output path must never overwrite source code or local credentials.
    if output.exists() or output.suffix != ".json":
        parser.error("--output должен указывать на новый файл с расширением .json")
    report = {
        "created_at": stamp, "prompt": PROMPT, "schema": SCHEMA,
        "runs_per_mode": args.runs, "max_tokens": args.max_tokens,
        "completed": False, "records": [],
    }
    save_report(output, report)
    abort = False
    try:
        for mode in MODES:
            for run in range(1, args.runs + 1):
                payload = build_payload(mode, args.max_tokens)
                record = {"mode": mode, "run": run, "request": payload}
                try:
                    record.update(inspect_answer(send_request(payload)))
                except HTTPError as error:
                    record["error"] = {"kind": "http", "status": error.code}
                    try:
                        detail = json.loads(error.read()).get("error", {})
                        detail = detail.get("message", "") if isinstance(detail, dict) else str(detail)
                        for secret in (API_KEY, API_URL, urlsplit(API_URL).hostname):
                            if secret:
                                detail = detail.replace(secret, "[hidden]")
                        record["error"]["message"] = detail[:500]
                    except (ValueError, AttributeError, TypeError):
                        pass
                    abort = error.code in (401, 403, 429)
                except (URLError, TimeoutError) as error:
                    record["error"] = {
                        "kind": "network", "reason_type": type(getattr(error, "reason", error)).__name__,
                        "message": "Нет соединения с API или истёк таймаут.",
                    }
                    abort = True
                except (ValueError, KeyError, IndexError, TypeError):
                    record["error"] = {"kind": "response", "message": "Неожиданный формат ответа API."}
                report["records"].append(record)
                save_report(output, report)
                print(f"\n[{mode} {run}/{args.runs}]")
                print(record.get("content") or record.get("error") or "Пустой ответ")
                if record.get("validation_error"):
                    print(record["validation_error"])
                if abort:
                    break
            if abort:
                break
        report["completed"] = not abort
    except KeyboardInterrupt:
        report["interrupted"] = True
        abort = True
    save_report(output, report)
    print("\nСравнение:")
    print("\n".join(report["conclusions"]))
    if abort:
        print("Эксперимент прерван; сравнение всех режимов не выполнено.")
    print(f"\nРезультаты: {output}")
    return 2 if abort or any("error" in row for row in report["records"]) else 0


if __name__ == "__main__":
    sys.exit(main())
