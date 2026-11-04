# Минимальный LLM CLI

Один запрос из консоли → POST в API → текст ответа в консоли.
Ожидается OpenAI-совместимый API: Bearer-ключ, `model`, `messages`
и ответ в `choices[0].message.content`.
Совместимость конкретного WB API пока не подтверждена.

## Запуск

```bash
python -m pip install -r requirements.txt
python main.py
```

Перед запуском заполните `.env` рядом с `main.py`:

```dotenv
WB_API_KEY=ваш_ключ
WB_API_URL=полный_URL_для_chat_completions
WB_MODEL=точный_ID_модели
```

Ключ не включён в проект: добавьте свой локально. `.env` исключён из Git.
При свежем клонировании создайте `.env` рядом с `main.py`.
Переменные окружения имеют приоритет над `.env`.
# vibe-coding
