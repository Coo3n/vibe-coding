"""Локальный эксперимент: 4 независимых промпта × 2 уровня рассуждения."""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


TASKS = [
    ("logic_4", "Трое жителей острова — А, Б и В — либо всегда говорят правду, либо всегда лгут. "
     "А говорит: «Б лжёт». Б говорит: «А и В одного типа (оба правдивы или оба лгут)». "
     "В говорит: «Б говорит правду». Кто из них говорит правду?\n"
     "A) только А  B) только Б  C) А и В  D) Б и В", "A"),
    ("logic_5", "В одном из четырёх конвертов A, B, C, D лежит приз. Надписи на конвертах: "
     "A: «Приз в B или C»; B: «Приз не в A»; C: «Приз в D»; D: «Приз не в B». "
     "Ровно две надписи истинны. В каком конверте приз?\n"
     "A) A  B) B  C) C  D) D", "B"),
    ("logic_6", "Трое жителей острова — А, Б и В — либо всегда говорят правду, либо всегда лгут. "
     "А говорит: «Б и В оба лгут». Б говорит: «А лжёт». В говорит: «Б лжёт». "
     "Кто говорит правду? Выберите единственный вариант, согласованный со всеми тремя высказываниями.\n"
     "A) только А  B) А и В  C) только Б  D) только В", "C"),
    ("analysis_4", "В урне 4 красных и 3 синих шара. Случайно без возвращения вынимают два шара. "
     "Какова вероятность того, что оба шара одного цвета? Считайте все пары шаров "
     "равновероятными; порядок извлечения не важен.\n"
     "A) 1/2  B) 3/7  C) 4/7  D) 9/49", "B"),
    ("analysis_5", "Одна труба наполняет пустой бак за 6 часов, вторая — за 8 часов, "
     "а открытый слив опорожняет полный бак за 12 часов. Сначала открыты обе трубы "
     "и слив; через 2 часа слив закрывают. Сколько часов пройдёт от начала до наполнения "
     "бака? Скорости постоянны и суммируются.\n"
     "A) 3 часа  B) 4 часа  C) 4,5 часа  D) 5 часов", "B"),
    ("analysis_6", "Код — трёхзначное десятичное число без повторяющихся цифр: "
     "первая цифра не может быть нулём. Сколько таких кодов делятся на 5? "
     "Рассмотрите отдельно случаи последней цифры 0 и 5; число 0 как средняя "
     "цифра допустимо.\nA) 144  B) 128  C) 152  D) 136", "D"),
]
METHODS = ("direct", "steps", "self_prompt", "experts")
LABELS = {"direct": "Прямой", "steps": "Пошагово", "self_prompt": "Самопромпт", "experts": "Эксперты"}


def should_review(force, skip, is_tty):
    return force or (is_tty and not skip)


def render_request(body, index):
    settings = {key: value for key, value in body.items() if key not in ("model", "messages")}
    lines = [f"╭─ Вызов {index:02d} · отправляем в API",
             f"│ Параметр model (маршрут API): {body['model']}",
             f"│ Рассуждение: {json.dumps(settings, ensure_ascii=False)}", "│ Промпт:"]
    for message in body["messages"]:
        lines.append(f"│   [{message['role']}]")
        lines.extend(f"│   {line}" for line in message["content"].splitlines())
    return "\n".join([*lines, "╰" + "─" * 48])


def render_response(result):
    usage = result["usage"]
    reasoning = reported_reasoning_tokens(usage)
    if reasoning is None:
        reasoning = "—"
    return "\n".join([
        "╭─ Ответ API",
        f"│ Фактическая модель (ответ API): {result.get('response_model') or '—'} "
        f"· завершение: {result.get('finish_reason') or '—'}",
        f"│ Токены: вход {usage.get('prompt_tokens', '—')} · выход {usage.get('completion_tokens', '—')} "
        f"(из них рассуждение {reasoning}) · всего {usage['total_tokens']}",
        f"│ Время: {result['seconds']:.2f} с", "│ Ответ:",
        *(f"│   {line}" for line in result["text"].splitlines()),
        "╰" + "─" * 48,
    ])


def build_messages(method, task):
    if method == "direct":
        prompt = task
    elif method == "steps":
        prompt = task + "\n\nрешай пошагово."
    elif method == "experts":
        prompt = (task + "\n\nРешите задачу независимо друг от друга как три эксперта: "
                  "аналитик, инженер и критик. Запишите отдельное решение каждого "
                  "на строке, начинающейся с «Аналитик:», «Инженер:», «Критик:». "
                  "Завершите каждое решение записью «Ответ: X». "
                  "Не согласовывайте ответы между собой.")
    elif method == "self_prompt":
        prompt = ("Составь промпт, который поможет другой модели решить задачу ниже. "
                  "Включи в промпт точный текст задачи. Верни только готовый промпт, "
                  "не решай саму задачу.\n\n" + task)
    else:
        raise ValueError(f"Неизвестный способ: {method}")
    return [{"role": "user", "content": prompt}]


def _score_one(task_id, text):
    key = next(answer for name, _, answer in TASKS if name == task_id)
    letters = "A-DАБ"
    # Берём последний явно обозначенный ответ: модель может исправить себя в том же сообщении.
    matches = list(re.finditer(rf"(?i)\b(?:ответ\s*[:—-]|вариант\s*[—:-])\s*\**\s*([{letters}])(?=\s*(?:\)|[.!?*]|$))", text))
    matches += list(re.finditer(rf"\\boxed\s*\{{\s*([{letters}])\s*\}}", text, re.IGNORECASE))
    if matches:
        letter = max(matches, key=lambda m: m.start()).group(1).upper().translate(str.maketrans({"А": "A", "Б": "B"}))
        return letter == key
    match = re.search(rf"(?im)^\s*\**\s*([{letters}])\s*\)", text)
    if not match:
        match = re.fullmatch(rf"\s*([{letters}])\s*[.!]?\s*", text, re.IGNORECASE)
    if match:
        return match.group(1).upper().translate(str.maketrans({"А": "A", "Б": "B"})) == key
    options = {letter.upper().translate(str.maketrans({"А": "A", "Б": "B"}))
               for letter in re.findall(rf"(?i)(?<!\w)([{letters}])\s*\)", text)}
    if options:
        return options == {key}
    answer = re.search(r"(?i)\bответ\s*:\s*([^\n.]+)", text)
    if answer:
        fragment = answer.group(1).lower()
        expected = {"analysis_4": "3/7", "analysis_5": "4", "analysis_6": "136"}.get(task_id)
        if expected:
            return re.findall(r"(?<!\d)\d+(?:/\d+)?(?:[.,]\d+)?(?!\d)", fragment) == [expected]
    return False


def evaluate(task_id, text, method):
    if method != "experts":
        return _score_one(task_id, text)
    sections = re.split(r"(?im)^\s*(аналитик|инженер|критик)\s*:", text)
    answers = {sections[i].lower(): sections[i + 1] for i in range(1, len(sections) - 1, 2)}
    return all(name in answers and _score_one(task_id, answers[name])
               for name in ("аналитик", "инженер", "критик"))


def run_case(api, method, effort, task):
    task_id, text, _ = task
    messages = build_messages(method, text)
    stages = []
    first = api(messages, effort)
    stages.append({"messages": messages, **first})
    if method == "self_prompt":
        generated = first["text"].strip()
        if not generated:
            raise ValueError("Модель вернула пустой промпт")
        messages = [{"role": "user", "content": generated}]
        second = api(messages, effort)
        stages.append({"messages": messages, **second})
    else:
        second = first
    return {"task": task_id, "method": method, "effort": effort,
            "answer": second["text"], "correct": evaluate(task_id, second["text"], method),
            "tokens": sum(stage["tokens"] for stage in stages),
            "seconds": sum(stage["seconds"] for stage in stages), "stages": stages}


def load_env(path):
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip("\"'")
    for key in ("LLM_API_KEY", "LLM_API_URL", "LLM_MODEL"):
        if not config.get(key):
            raise ValueError(f"Нет {key} в {path}")
    return config


def reported_reasoning_tokens(usage):
    value = usage.get("reasoning_tokens")
    if value is None:
        value = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return value


def make_api(config, reasoning_field, low_budget=1024, on_request=None, on_response=None):
    if reasoning_field == "budget_tokens":
        if low_budget < 0:
            raise ValueError("Для budget_tokens укажите --low-budget >= 0")
    resolved_model = None

    def call(messages, effort):
        nonlocal resolved_model
        body = {"model": config["LLM_MODEL"], "messages": messages}
        if reasoning_field == "reasoning_effort" or (reasoning_field == "budget_tokens" and effort == "high"):
            body["reasoning_effort"] = effort
        else:
            body["reasoning"] = {"budget_tokens": low_budget}
        if on_request:
            on_request(body)
        request = urllib.request.Request(
            config["LLM_API_URL"], data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + config["LLM_API_KEY"],
                     "Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            error.close()
            raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
        seconds = time.perf_counter() - start
        response_model = data.get("model")
        if not response_model:
            raise ValueError("API не вернул модель: нельзя подтвердить единый ID в эксперименте")
        if "/" in config["LLM_MODEL"] and response_model != config["LLM_MODEL"]:
            raise ValueError(f"Запрошена модель {config['LLM_MODEL']}, API вернул модель {response_model!r}")
        if resolved_model and response_model and response_model != resolved_model:
            raise ValueError(f"API сменил модель во время эксперимента: {resolved_model} → {response_model}")
        resolved_model = response_model
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Нет текстового ответа модели: {str(message)[:300]}")
        usage = data.get("usage") or {}
        if "total_tokens" not in usage:
            raise ValueError("API не вернул usage.total_tokens; сравнение токенов невозможно")
        if reasoning_field == "budget_tokens" and effort == "low":
            reasoning_tokens = reported_reasoning_tokens(usage)
            if reasoning_tokens is None:
                raise ValueError("API не вернул reasoning_tokens: бюджет невозможно проверить")
            if reasoning_tokens > low_budget:
                raise ValueError(f"API потратил {reasoning_tokens} токенов рассуждения при бюджете {low_budget}")
        result = {"text": content, "tokens": usage["total_tokens"],
                  "usage": usage, "seconds": seconds,
                  "finish_reason": choice.get("finish_reason"),
                  "response_model": response_model,
                  "request_params": {key: value for key, value in body.items() if key != "messages"}}
        if on_response:
            on_response(result)
        return result
    return call


def table(records):
    lines = ["| Способ | Рассуждение | Верно | Токены | Время, с |",
             "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for effort in ("low", "high"):
            rows = [r for r in records if r["method"] == method and r["effort"] == effort]
            lines.append(f"| {LABELS[method]} | {effort} | "
                         f"{sum(r['correct'] for r in rows)}/{len(TASKS)} | "
                         f"{sum(r['tokens'] for r in rows)} | "
                         f"{sum(r['seconds'] for r in rows):.2f} |")
    return "\n".join(lines)


def console_table(records):
    columns = ("Способ", "Уровень", "Верно", "Токены", "Время")
    rows = []
    for method in METHODS:
        for effort in ("low", "high"):
            group = [r for r in records if (r["method"], r["effort"]) == (method, effort)]
            rows.append((LABELS[method], effort,
                         f"{sum(r['correct'] for r in group)}/{len(TASKS)}",
                         str(sum(r["tokens"] for r in group)),
                         f"{sum(r['seconds'] for r in group):.2f} с"))
    widths = [max(len(row[i]) for row in [columns, *rows]) for i in range(len(columns))]

    def border(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def line(row):
        return "│ " + " │ ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " │"

    return "\n".join([border("┌", "┬", "┐"), line(columns),
                      border("├", "┼", "┤"), *(line(row) for row in rows),
                      border("└", "┴", "┘")])


def conclusion(records):
    def summary(method, effort):
        rows = [r for r in records if r["method"] == method and r["effort"] == effort]
        return sum(r["correct"] for r in rows), sum(r["tokens"] for r in rows), sum(r["seconds"] for r in rows)

    lines = []
    for effort in ("low", "high"):
        baseline = summary("direct", effort)
        better = []
        equal = []
        worse = []
        for method in METHODS[1:]:
            score = summary(method, effort)
            detail = f"{LABELS[method]} ({score[0]}/{len(TASKS)}, {score[1]} токенов, {score[2]:.2f} с)"
            (better if score[0] > baseline[0] else equal if score[0] == baseline[0] else worse).append(detail)
        lines.append(f"При {effort} прямой запрос: {baseline[0]}/{len(TASKS)}, "
                     f"{baseline[1]} токенов, {baseline[2]:.2f} с.")
        if effort == "high" and not better:
            lines.append("При high ни один приём не улучшил точность относительно прямого запроса на этих задачах.")
        for label, methods in (("Лучше по точности", better), ("Та же точность", equal),
                               ("Хуже по точности", worse)):
            if methods:
                lines.append(label + ": " + "; ".join(methods) + ".")
    lines.append("Это наблюдение на шести задачах, а не доказательство бесполезности приёмов; "
                 "сравнивайте также токены, время и полные ответы.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reasoning-field", choices=("reasoning_effort", "budget_tokens"),
                         default="budget_tokens")
    parser.add_argument("--low-budget", type=int, default=1024,
                        help="Запрашиваемый предел токенов рассуждения (по умолчанию 1024)")
    review_flags = parser.add_mutually_exclusive_group()
    review_flags.add_argument("--review", action="store_true", help="Ручная проверка даже без TTY")
    review_flags.add_argument("--no-review", action="store_true", help="Без вопросов, только автооценка")
    args = parser.parse_args()
    args.output = args.output or Path("results/experiment.json")
    if args.output.suffix != ".json" or args.output.resolve() == args.env.resolve():
        parser.error("--output должен быть JSON-файлом и не может совпадать с .env")
    config = load_env(args.env)
    review = should_review(args.review, args.no_review, sys.stdin.isatty())
    call_count = 0

    def show_request(body):
        nonlocal call_count
        call_count += 1
        print(render_request(body, call_count), flush=True)

    def show_response(result):
        print(render_response(result), flush=True)

    try:
        api = make_api(config, args.reasoning_field, args.low_budget,
                       on_request=show_request, on_response=show_response)
    except ValueError as error:
        parser.error(str(error))
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"model": config["LLM_MODEL"], "requested_model": config["LLM_MODEL"],
                "reasoning_field": args.reasoning_field,
                "efforts": ["low", "high"], "started": datetime.now().astimezone().isoformat(),
                "completed": False}
    if args.reasoning_field == "budget_tokens":
        metadata.update(low_budget=args.low_budget, high_effort="high",
                        budget_status="observed_within_limit", zero_reasoning_calls=0)
    for method in METHODS:
        for effort in ("low", "high"):
            for task in TASKS:
                print(f"\n━━━ {len(records) + 1:02d}/{len(TASKS) * len(METHODS) * 2} "
                      f"· {LABELS[method]} · {effort} · {task[0]} ━━━", flush=True)
                record = run_case(api, method, effort, task)
                response_model = record["stages"][-1].get("response_model")
                if response_model:
                    metadata["model"] = metadata["resolved_model"] = response_model
                if args.reasoning_field == "budget_tokens" and effort == "low":
                    previous_zero = metadata["zero_reasoning_calls"]
                    metadata["zero_reasoning_calls"] += sum(
                        reported_reasoning_tokens(stage["usage"]) == 0
                        for stage in record["stages"])
                    if metadata["zero_reasoning_calls"]:
                        metadata["budget_status"] = "unverified"
                        if not previous_zero:
                            print("Внимание: API сообщил 0 токенов рассуждения при low; "
                                  "бюджет не подтверждён. Прогон продолжается.")
                print(f"Проверка: {'✓ верно' if record['correct'] else '✗ не распознано или неверно'} "
                      f"· эталон: {task[2]}")
                if review:
                    choice = input("Оценка [Enter = оставить, 1 = верно, 0 = неверно]: ").strip()
                    if choice in ("0", "1"):
                        record["correct"] = choice == "1"
                records.append(record)
    metadata["completed"] = True
    args.output.write_text(json.dumps({"metadata": metadata, "records": records},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = table(records)
    finding = conclusion(records)
    if args.reasoning_field == "budget_tokens":
        field = "reasoning.budget_tokens / reasoning_effort"
        levels = f"{args.low_budget} / high"
    else:
        field = "reasoning_effort"
        levels = "low / high"
    print(f"\nМодель: {metadata['model']} · Параметр API: {field} = {levels}")
    budget_note = ""
    if args.reasoning_field == "budget_tokens":
        budget_note = (f"Бюджет не подтверждён: API сообщил 0 токенов рассуждения в "
                       f"{metadata['zero_reasoning_calls']} low-вызовах. "
                       "Результаты low — наблюдение, не доказательство действия лимита."
                       if metadata["zero_reasoning_calls"] else
                       "Расход рассуждения не превышал бюджет; действие лимита без подтверждения API не доказано.")
        print(budget_note)
    print(console_table(records) + "\n\nВывод:\n" + finding)
    md_path = args.output.with_suffix(".md")
    md_path.write_text(f"# Сравнение промптов\n\nМодель: {metadata['model']} · "
                       f"параметр API: {field} = {levels}\n\n" +
                       (f"**{budget_note}**\n\n" if budget_note else "") + markdown +
                       "\n\n## Вывод\n\n" + finding +
                       "\n\nВремя — сумма длительностей API-вызовов; токены — usage.total_tokens. "
                       "Самопромпт включает оба вызова. Эксперты засчитываются, только если верны все три.\n",
                       encoding="utf-8")
    print(f"\nПодробности: {args.output}; таблица: {md_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
