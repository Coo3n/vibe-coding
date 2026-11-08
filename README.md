# vibe-coding

Практические эксперименты с LLM: от первого API-запроса до сравнения моделей.
Ветка `main` — оглавление; код, инструкции и результаты каждого задания находятся в своей ветке.

## Задания

| День | Задание и код | Что исследуем | Результаты | Видео |
|---|---|---|---|---|
| 1 | [Первый запрос к LLM](https://github.com/Coo3n/vibe-coding/tree/task-1) | Минимальный Python CLI, API и вывод ответа | В консоли | [Запись](https://github.com/Coo3n/vibe-coding/blob/task-1/results/task-1-demo.mov) |
| 2 | [Контроль ответа](https://github.com/Coo3n/vibe-coding/tree/task-2) | JSON, лимит токенов, stop sequence и JSON Schema | [Данные](https://github.com/Coo3n/vibe-coding/blob/task-2/results/20261107T165101807474Z.json) | [Запись](https://github.com/Coo3n/vibe-coding/blob/task-2/results/demo.mov) |
| 3 | [Способы построения промптов](https://github.com/Coo3n/vibe-coding/tree/task-3) | Четыре способа решения задач и два режима рассуждения | [Отчёт](https://github.com/Coo3n/vibe-coding/blob/task-3/results/experiment.md) | [Запись](https://github.com/Coo3n/vibe-coding/blob/task-3/results/final-experiment-demo.mp4) |
| 4 | [Температура модели](https://github.com/Coo3n/vibe-coding/tree/task-4) | Точность, формат и разнообразие при 0 / 0.7 / 1.2 | [Отчёт](https://github.com/Coo3n/vibe-coding/blob/task-4/results/temperature-glm.md) | [Запись](https://github.com/Coo3n/vibe-coding/blob/task-4/results/demo.mp4) |
| 5 | [Сравнение моделей](https://github.com/Coo3n/vibe-coding/tree/task-5) | Qwen, DeepSeek и GLM: 10 задач, время, токены и расчётная стоимость | [Medium](https://github.com/Coo3n/vibe-coding/blob/task-5/results/day5-medium-10/comparison.md) · [High](https://github.com/Coo3n/vibe-coding/blob/task-5/results/day5-high-10/comparison.md) | [Запись](https://github.com/Coo3n/vibe-coding/blob/task-5/results/demo.mp4) |

## Как открыть задание

```bash
git fetch origin
git switch task-1
```

Замени `task-1` на нужную ветку: `task-2`, `task-3`, `task-4` или `task-5`.
Команды запуска и настройки `.env` — в README выбранной ветки. API-ключи в репозиторий не добавляются.
