"""Профиль пользователя: структурированные предпочтения поверх слоёв памяти.

Профиль — отдельная сущность: стиль (кратко/подробно), формат (с кодом/без
кода), тон (формально/неформально) плюс свободные ограничения. Хранится в
data/profiles/<имя>.json, инжектится в КАЖДЫЙ запрос как system-блок после
long-term (см. MemoryStore.context). Пополняется двумя путями:
  1) автоматически — служебный LLM-вызов после реплики пользователя извлекает
     высказывания-предпочтения (расход считается честно: side_calls/side_tokens);
  2) вручную — команды чата /prefer и /profile.
"""

import fcntl
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


# Схема профиля: поле → допустимые значения.
PREFERENCE_FIELDS = ('стиль', 'формат', 'тон')
ATTRIBUTES = dict(zip(PREFERENCE_FIELDS, ('style', 'code_format', 'tone')))
FIELD_VALUES = {
    'стиль': ('кратко', 'подробно'),
    'формат': ('с кодом', 'без кода'),
    'тон': ('формально', 'неформально'),
}
# Псевдонимы полей для ручной команды /prefer.
FIELD_ALIASES = {
    'стиль': ('стиль', 'style', 'длина', 'длину', 'подробность'),
    'формат': ('формат', 'код', 'code', 'format'),
    'тон': ('тон', 'tone', 'обращение'),
}

PROFILE_HEADER = 'Профиль пользователя и предпочтения'
SHORT_MAX_WORDS = 90
DETAILED_MIN_WORDS = 150

LENGTH_INSTRUCTIONS = {
    'кратко': f'Уложись в {SHORT_MAX_WORDS} слов во всём ответе. Начни с сути и оставь только нужные детали.',
    'подробно': f'Напиши не менее {DETAILED_MIN_WORDS} слов. Объясни ход решения и дай уместные детали.',
}
TONE_INSTRUCTIONS = {
    'формально': 'Обращайся к собеседнику на «Вы» с прописной буквы. '
                 'Используй естественное обращение, например «Вы можете» или «проверьте». '
                 'Пиши вежливо, без фамильярности, просторечий и эмодзи.',
    'неформально': 'Обращайся к собеседнику на «ты». '
                   'Используй естественное обращение, например «тебе поможет» или «проверь». '
                   'Пиши просто и дружелюбно.',
}

# Правило для извлечения предпочтений: служебный LLM-вызов после реплики.
PROFILE_SYSTEM_PROMPT = (
    'Ты — извлекатель предпочтений пользователя. Тебе дают текущий профиль '
    '(JSON) и новое сообщение пользователя. Найди в сообщении ЯВНЫЕ '
    'установки-предпочтения о том, как ассистент должен отвечать: длина '
    '(кратко/подробно), код (с кодом/без кода), тон (формально/неформально), '
    'а также любые другие ограничения («без эмодзи», «только по делу», '
    '«пишите списками»). Это установки НА БУДУЩЕЕ («теперь отвечай», '
    '«отныне всегда», «не используй больше», «перейди на Вы»). Разовые '
    'просьбы внутри конкретного вопроса («покажи кодом», «расскажи подробно '
    'про Гион») профиль НЕ меняют. Если предпочтений нет — верни профиль '
    'без изменений. Верни профиль как чистый JSON без markdown и пояснений, '
    'схема: {"стиль": "кратко|подробно", "формат": "с кодом|без кода", '
    '"тон": "формально|неформально", "ограничения": ["..."]}. '
    'Список ограничений дополняй новыми, дубликаты не создавай.')


class UserProfile:
    """Структурированные предпочтения одного пользователя.

    side_calls/side_tokens — расход служебных вызовов извлечения
    предпочтений (считается честно, как у working-памяти в Днях 10–11).
    """

    def __init__(self, name, style, code_format, tone, restrictions=None, path=None):
        for field, value in (('стиль', style), ('формат', code_format), ('тон', tone)):
            if value not in FIELD_VALUES[field]:
                raise ValueError(f'{field}: допустимо {"/".join(FIELD_VALUES[field])}, '
                                 f'получено «{value}».')
        self.name = name
        self.style = style
        self.code_format = code_format
        self.tone = tone
        if restrictions is None:
            restrictions = []
        if not isinstance(restrictions, list) or any(
                not isinstance(item, str) or not item.strip() for item in restrictions):
            raise ValueError('Ограничения профиля должны быть списком непустых строк.')
        self.restrictions = list(dict.fromkeys(' '.join(item.split()) for item in restrictions))
        self.path = Path(path) if path else None
        self.side_calls = 0
        self.side_tokens = 0

    # ---- диск: загрузка и сохранение ----

    @classmethod
    def from_dict(cls, data, path=None):
        return cls(data['имя'], data['стиль'], data['формат'], data['тон'],
                   data.get('ограничения'), path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f'Профиля нет: {path}. Доступны: '
                             f'{", ".join(p.stem for p in sorted(path.parent.glob("*.json"))) or "—"}')
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')), path)

    def to_dict(self):
        return {'имя': self.name, 'стиль': self.style, 'формат': self.code_format,
                'тон': self.tone, 'ограничения': list(self.restrictions)}

    def save(self):
        if self.path is None:
            raise ValueError('У профиля не задан путь для сохранения.')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                    delete=False) as file:
                temporary = Path(file.name)
                json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
                file.write('\n')
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # ---- блок для промпта: инжектится в каждый запрос ----

    def block(self, enforce=True):
        lines = [f'{PROFILE_HEADER} (активный профиль: {self.name}):',
                 f'- стиль ответов: {self.style}',
                 f'- формат: {self.code_format}',
                 f'- тон: {self.tone}']
        lines += [f'- ограничение: {item}' for item in self.restrictions]
        if enforce:
            lines.extend((LENGTH_INSTRUCTIONS[self.style], TONE_INSTRUCTIONS[self.tone]))
            lines.append('Соблюдай эти предпочтения в каждом ответе, включая текущий.')
        return '\n'.join(lines)

    def describe(self):
        text = (f'Стиль: {self.style} · формат: {self.code_format} · тон: {self.tone}')
        if self.restrictions:
            text += '\nОграничения: ' + '; '.join(self.restrictions)
        if self.path is not None:
            text += f'\nФайл: {self.path}'
        return text

    # ---- ручное редактирование: /prefer поле: значение ----

    def set(self, field, value):
        field = field.strip().lower()
        value = ' '.join(value.split())
        if field in ('ограничение', 'ограничения', 'restriction'):
            if not value:
                raise ValueError('Укажите текст ограничения.')
            return '\n'.join(self._apply({}, [value])) or 'Ограничение уже сохранено.'
        for canonical, aliases in FIELD_ALIASES.items():
            if field in aliases:
                field = canonical
                break
        else:
            raise ValueError('Поле: ' + '/'.join(PREFERENCE_FIELDS) + '/ограничения')
        value = value.lower()
        if value not in FIELD_VALUES[field]:
            raise ValueError(f'{field}: допустимо {"/".join(FIELD_VALUES[field])}.')
        return '\n'.join(self._apply({field: value}, [])) or 'Значение уже сохранено.'

    def _apply(self, fields, restrictions):
        if self.path is None:
            raise ValueError('У профиля не задан путь для сохранения.')
        lock_path = self.path.with_suffix(self.path.suffix + '.lock')
        with lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            profile = self.load(self.path)
            changes = []
            for field, value in fields.items():
                attribute = ATTRIBUTES[field]
                old = getattr(profile, attribute)
                if old != value:
                    setattr(profile, attribute, value)
                    changes.append(f'{field}: {old} → {value}')
            for item in restrictions:
                if item not in profile.restrictions:
                    profile.restrictions.append(item)
                    changes.append(f'ограничения += «{item}»')
            if changes:
                profile.save()
        self.style, self.code_format, self.tone = profile.style, profile.code_format, profile.tone
        self.restrictions = profile.restrictions
        return changes

    # ---- пополнение из диалога: служебный LLM-вызов ----

    def update_from_dialogue(self, user_text, chat):
        """Извлекает предпочтения из реплики; возвращает список изменений.

        Список пуст, если модель не нашла установок на будущее. Расход
        вызова засчитывается всегда — даже когда профиль не изменился.
        """
        before = self.to_dict()
        prompt = [
            {'role': 'system', 'content': PROFILE_SYSTEM_PROMPT},
            {'role': 'system', 'content': self.block(enforce=False)},
            {'role': 'user', 'content':
             f'Текущий профиль:\n{json.dumps(before, ensure_ascii=False)}\n\n'
             f'Новое сообщение пользователя:\n{user_text}'},
        ]
        text, usage = chat(prompt)
        if usage is not None:
            self.side_calls += 1
            self.side_tokens += usage['total_tokens']
        try:
            payload = text.strip()
            if payload.startswith('```') and payload.endswith('```'):
                payload = '\n'.join(payload.splitlines()[1:-1])
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError('не словарь')
        except (ValueError, KeyError, TypeError):
            return []  # ответ не разобран — профиль не трогаем, расход честно учтён
        fields = {field: data[field] for field in PREFERENCE_FIELDS
                  if data.get(field) in FIELD_VALUES[field] and data[field] != before[field]}
        restrictions = data.get('ограничения', [])
        additions = ([' '.join(item.split()) for item in restrictions
                      if isinstance(item, str) and item.strip()] if isinstance(restrictions, list)
                     else [])
        if not fields and not additions:
            return []
        return self._apply(fields, additions)


def seed_profiles(directory, seeds, overwrite=True):
    """Создаёт профили на диске (как сид long-term).

    overwrite=True — перезаписать сиды (эксперимент начинает чисто);
    False — не трогать существующие файлы (чат работает с накопленными).
    Возвращает словарь имя → загруженный профиль.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for seed in seeds:
        path = directory / (seed['файл'] + '.json')
        with path.with_suffix('.json.lock').open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if overwrite or not path.exists():
                UserProfile.from_dict(seed, path).save()
        profiles[seed['имя']] = UserProfile.load(path)
    return profiles
