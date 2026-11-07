# Минимальный LLM CLI

Один запрос из консоли → POST в API → текст ответа в консоли.
Ожидается OpenAI-совместимый API: Bearer-ключ, `model`, `messages`
и ответ в `choices[0].message.content`.
Провайдер, модель и ключ задаются через `.env`, без изменений кода.

## Task 1

🎬 [Посмотреть видео демонстрации](results/task-1-demo.mov)

## Запуск

```bash
python -m pip install -r requirements.txt
python main.py
```

Перед запуском заполните `.env` рядом с `main.py`:

```dotenv
LLM_API_KEY=ваш_ключ
LLM_API_URL=https://api.example.com/v1/chat/completions
LLM_MODEL=идентификатор_модели
```

Ключ не включён в проект: добавьте свой локально. `.env` исключён из Git.
При свежем клонировании создайте `.env` рядом с `main.py`.
Переменные окружения имеют приоритет над `.env`.
