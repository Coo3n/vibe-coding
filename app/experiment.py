"""Live full/compressed dialogues with matching user prompts and independent memory probes."""

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
    live = report.get('live_compression', False)
    dialogues = {mode: [trace for turn in report['dialogue'] if turn.get('mode') == mode
                        for trace in turn['trace']] for mode in ('full', 'compressed')}
    compression = [trace for event in report['compressions'] for trace in event['trace']]
    probes = {mode: [trace for probe in report['probes'][mode] for trace in probe['trace']]
              for mode in ('full', 'compressed')}
    full_cost = (dialogues['full'] if live else []) + probes['full']
    compressed_cost = (dialogues['compressed'] if live else []) + compression + probes['compressed']
    usage = {'preparation': usage_total(preparation), 'summary': usage_total(compression),
             'full_dialogue': usage_total(dialogues['full']),
             'compressed_dialogue': usage_total(dialogues['compressed']),
             'full_probes': usage_total(probes['full']),
             'compressed_probes': usage_total(probes['compressed']),
             'full_comparison': usage_total(full_cost),
             'compressed_comparison': usage_total(compressed_cost),
             'full_total': usage_total(full_cost if live else preparation + probes['full']),
             'compressed_total': usage_total(compressed_cost if live else preparation + compressed_cost),
             'actual_run': usage_total(preparation + compression + probes['full'] + probes['compressed'])}
    report['usage'] = usage
    report['score'] = {mode: {'correct': sum(p.get('evaluation', {}).get('correct', False)
                                           for p in report['probes'][mode]),
                             'checked': sum('evaluation' in p for p in report['probes'][mode]),
                             'attempted': len(report['probes'][mode]), 'total': len(FACTS)}
                       for mode in ('full', 'compressed')}
    comparable = (report.get('case', 'both') == 'both' and
                  all(report['score'][mode]['checked'] == len(FACTS) for mode in ('full', 'compressed')))
    report['savings'] = (savings(usage['full_comparison']['total_tokens'],
                                 usage['compressed_comparison']['total_tokens'])
                         if comparable else savings(None, None))
    full = {p['fact_id']: p.get('evaluation', {}) for p in report['probes']['full']}
    compact = {p['fact_id']: p.get('evaluation', {}) for p in report['probes']['compressed']}
    report['lost_after_compression'] = [f['id'] for f in FACTS
                                       if full.get(f['id'], {}).get('correct') and
                                       compact.get(f['id']) and not compact[f['id']].get('correct')]


def run_experiment(client, directory, on_event=None, keep_last=8, compress_every=20, turns=50, case='both'):
    if type(turns) is not int or turns not in (15, 30, 50):
        raise ValueError('Для эксперимента выберите 15, 30 или 50 ходов.')
    if case not in ('full', 'compressed', 'both'):
        raise ValueError('Выберите full, compressed или both.')
    prompts = PROMPTS[:turns]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    on_event = on_event or (lambda event: None)
    report = {'version': 2, 'case': case, 'live_compression': True,
              'started': datetime.now().astimezone().isoformat(), 'status': 'running',
              'topic': 'Подготовка овощного онлайн-прилавка',
              'method': ('Два самостоятельных диалога с одинаковыми пользовательскими сообщениями. ' if case == 'both' else
                         f"Один самостоятельный диалог {'без сжатия' if case == 'full' else 'со сжатием'}. ") +
                        f'В сжатом режиме summary создаётся прямо после каждых {compress_every // 2} ходов. '
                        'Ответы модели могут различаться. Сравнивается полный расход диалога, summary и проверок.',
              'settings': {'keep_last': keep_last, 'compress_every': compress_every, 'turns': turns},
              'facts': deepcopy(FACTS), 'dialogue': [], 'compressions': [], 'contexts': {},
              'probes': {'full': [], 'compressed': []}}

    def save():
        report['model'] = client.model_name
        if 'counts' in report:
            report['counts']['compressions'] = sum(event.get('summary') is not None for event in report['compressions'])
            report['counts']['compression_attempts'] = len(report['compressions'])
        summarize(report)
        atomic_json(directory / 'report.json', report)

    save()  # Record the answer key before generating any dialogue.
    try:
        report['counts'] = {'visible_messages': len(prompts) * 2,
                            'full_records': None,
                            'compressed_tail_records': None,
                            'compressed_tail_visible_messages': None, 'compressions': 0}
        save()
        modes = ('full', 'compressed') if case == 'both' else (case,)
        for mode in modes:
            on_event({'kind': 'case_start', 'mode': mode, 'live': True, 'turns': turns,
                      'every_turns': compress_every // 2})
            client.phase = 'dialogue_' + mode
            compression_events = []

            def agent_event(event):
                if event['kind'] == 'compression':
                    compression_events.append(deepcopy(event['event']))
                on_event(event)

            agent = Agent(client, Context(mode, keep_last, compress_every), on_event=agent_event)
            for number, prompt in enumerate(prompts, 1):
                compression_events.clear()
                on_event({'kind': 'dialogue', 'number': number, 'total': len(prompts), 'prompt': prompt})
                row = {'number': number, 'mode': mode, 'prompt': {'message': prompt}, 'trace': [], 'turn': None}
                report['dialogue'].append(row)
                try:
                    agent.ask(prompt)
                    row['turn'] = deepcopy(agent.last_turn)
                    on_event({'kind': 'answer', 'reply': agent.reply, 'trace': agent.trace})
                finally:
                    row['trace'] = deepcopy([t for t in agent.trace if t['kind'] != 'summary'])
                    for index, trace in enumerate(t for t in agent.trace if t['kind'] == 'summary'):
                        event = (compression_events[index] if index < len(compression_events) else
                                 {'after_visible_messages': number * 2, 'summary': None})
                        event['trace'] = [deepcopy(trace)]
                        report['compressions'].append(event)
                    save()
            context = agent.context
            report['contexts'][mode] = context.to_dict()
            if mode == 'compressed':
                report['counts'].update(compressed_tail_records=len(context.messages),
                                        compressed_tail_visible_messages=context.visible_messages,
                                        compressions=len(report['compressions']))
            else:
                report['counts']['full_records'] = len(context.messages)
            save()
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
            on_event({'kind': 'case_complete', 'mode': mode, 'report': report})
        by_mode = {mode: {message['tool_calls'][0]['function']['name']
                         for row in report['dialogue'] if row['mode'] == mode
                         for message in row['turn'] if message.get('tool_calls')} for mode in modes}
        tool_names = set().union(*by_mode.values())
        report['tool_calls_observed'] = sorted(tool_names)
        report['tool_calls_by_mode'] = {mode: sorted(names) for mode, names in by_mode.items()}
        report['status'] = 'completed' if all(names >= {'get_products', 'calculate_batch'} for names in by_mode.values()) else 'incomplete'
        if report['status'] == 'incomplete':
            report['error'] = 'Модель не вызвала оба инструмента в сценарии; см. трассировку.'
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error) or 'Прервано пользователем'
    finally:
        save()
    return report
