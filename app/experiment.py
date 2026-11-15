"""One real dialogue, two contexts, independent questions with no answer leakage."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
import unicodedata

from .agent import Agent
from .context import Context, atomic_json
from .protocol import dumps, invoke
from .scenario import PROMPTS, FACTS
from .tokens import usage_total, savings


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold().replace('ё', 'е')
    value = re.sub(r'[«»"“”]', '', value).strip().rstrip('.')
    value = re.sub(r'[‐‑–—]', '-', value)
    value = re.sub(r'\s+', ' ', value)
    if re.fullmatch(r'\d[\d ]*(?:[.,]0+)?', value):
        value = value.replace(' ', '').replace(',', '.')
        value = value.split('.')[0]
    return value


def evaluate(answer, fact):
    value = answer['value']
    correct = value is not None and normalize(value) in {normalize(s) for s in fact['accepted']}
    return {'status': 'correct' if correct else 'missing' if value is None else 'mismatch',
            'correct': correct, 'value': value}


def summarize(report):
    preparation = [trace for turn in report['dialogue'] for trace in turn['trace']]
    compression = [trace for event in report['compressions'] for trace in event['trace']]
    probes = {mode: [trace for probe in report['probes'][mode] for trace in probe['trace']]
              for mode in ('full', 'compressed')}
    usage = {'preparation': usage_total(preparation), 'summary': usage_total(compression),
             'full_probes': usage_total(probes['full']),
             'compressed_probes': usage_total(probes['compressed']),
             'full_comparison': usage_total(probes['full']),
             'compressed_comparison': usage_total(compression + probes['compressed']),
             'full_total': usage_total(preparation + probes['full']),
             'compressed_total': usage_total(preparation + compression + probes['compressed']),
             'actual_run': usage_total(preparation + compression + probes['full'] + probes['compressed'])}
    report['usage'] = usage
    report['savings'] = savings(usage['full_comparison']['total_tokens'],
                                usage['compressed_comparison']['total_tokens'])
    report['score'] = {mode: {'correct': sum(p.get('evaluation', {}).get('correct', False)
                                           for p in report['probes'][mode]),
                             'checked': sum('evaluation' in p for p in report['probes'][mode]),
                             'attempted': len(report['probes'][mode]), 'total': len(FACTS)}
                       for mode in ('full', 'compressed')}
    full = {p['fact_id']: p.get('evaluation', {}) for p in report['probes']['full']}
    compact = {p['fact_id']: p.get('evaluation', {}) for p in report['probes']['compressed']}
    report['lost_after_compression'] = [f['id'] for f in FACTS
                                       if full.get(f['id'], {}).get('correct') and
                                       compact.get(f['id']) and not compact[f['id']].get('correct')]


def run_experiment(client, directory, on_event=None, keep_last=8, compress_every=20, turns=50):
    if type(turns) is not int or turns not in (15, 30, 50):
        raise ValueError('Для эксперимента выберите 15, 30 или 50 ходов.')
    prompts = PROMPTS[:turns]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    on_event = on_event or (lambda event: None)
    report = {'version': 1, 'started': datetime.now().astimezone().isoformat(), 'status': 'running',
              'topic': 'Подготовка овощного онлайн-прилавка',
              'method': 'Общая реальная переписка; replay сжатия; независимые пробы. '
                        'Replay не измеряет генерацию второго диалога со сжатым контекстом.',
              'settings': {'keep_last': keep_last, 'compress_every': compress_every, 'turns': turns},
              'facts': deepcopy(FACTS), 'dialogue': [], 'compressions': [],
              'probes': {'full': [], 'compressed': []}}

    def save():
        report['model'] = client.model_name
        summarize(report)
        atomic_json(directory / 'report.json', report)

    save()  # Record the answer key before generating any dialogue.
    try:
        client.phase = 'dialogue'
        agent = Agent(client, Context(mode='full'), on_event=on_event)
        for number, prompt in enumerate(prompts, 1):
            on_event({'kind': 'dialogue', 'number': number, 'total': len(prompts), 'prompt': prompt})
            row = {'number': number, 'prompt': {'message': prompt}, 'trace': [], 'turn': None}
            report['dialogue'].append(row)
            try:
                agent.ask(prompt)
                row['turn'] = deepcopy(agent.last_turn)
                on_event({'kind': 'answer', 'reply': agent.reply, 'trace': agent.trace})
            finally:
                row['trace'] = deepcopy(agent.trace)
                save()

        client.phase = 'summary'
        compressed = Context(keep_last=keep_last, compress_every=compress_every)
        for number, row in enumerate(report['dialogue'], 1):
            compressed.append_turn(row['turn'])
            if compressed.visible_messages - keep_last >= compress_every:
                event = {'after_visible_messages': number * 2, 'trace': [], 'summary': None}
                report['compressions'].append(event)
                on_event({'kind': 'compression_start', 'after': number * 2})
                try:
                    compressed.compress(client, event['trace'])
                    event['summary'] = deepcopy(compressed.summary)
                    event['tail_visible_messages'] = compressed.visible_messages
                    on_event({'kind': 'compression', 'event': event})
                finally:
                    save()
        report['contexts'] = {'full': agent.context.to_dict(), 'compressed': compressed.to_dict()}
        report['counts'] = {'visible_messages': len(prompts) * 2,
                            'full_records': len(agent.context.messages),
                            'compressed_tail_records': len(compressed.messages),
                            'compressed_tail_visible_messages': compressed.visible_messages,
                            'compressions': len(report['compressions'])}
        save()
        for mode, context in (('full', agent.context), ('compressed', compressed)):
            client.phase = mode
            for number, fact in enumerate(FACTS, 1):
                on_event({'kind': 'probe_start', 'mode': mode, 'number': number, 'fact': fact})
                # No tools, no answer key, no other probes in this request.
                messages = context.request_messages() + [{
                    'role': 'user', 'content': dumps({'question': fact['question'],
                        'instruction': 'Проверка памяти. Верни только JSON с полем value: '
                                       'краткое точное значение из переписки или null. '
                                       'Не угадывай. Числа без единиц, даты YYYY-MM-DD.'})}]
                probe = {'fact_id': fact['id'], 'trace': []}
                report['probes'][mode].append(probe)
                try:
                    response = invoke(client, messages, 'probe', probe['trace'])
                    probe['answer'] = response
                    probe['evaluation'] = evaluate(response, fact)
                    on_event({'kind': 'probe', 'mode': mode, 'number': number, 'fact': fact, 'probe': probe})
                finally:
                    save()
        tool_names = {message['tool_calls'][0]['function']['name']
                      for row in report['dialogue'] for message in row['turn'] if message.get('tool_calls')}
        report['tool_calls_observed'] = sorted(tool_names)
        report['status'] = 'completed' if tool_names >= {'get_products', 'calculate_batch'} else 'incomplete'
        if report['status'] == 'incomplete':
            report['error'] = 'Модель не вызвала оба инструмента в сценарии; см. трассировку.'
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or 'Прервано пользователем'
    finally:
        save()
    return report
