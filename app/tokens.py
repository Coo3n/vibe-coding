"""Actual API usage only. Unknown is not zero; cache is already included."""


def usage_total(traces):
    result = {'requests': len(traces), 'unknown_requests': 0,
              'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    for trace in traces:
        usage = trace.get('usage')
        valid = isinstance(usage, dict) and all(
            type(usage.get(key)) is int and usage[key] >= 0
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        if not valid:
            result['unknown_requests'] += 1
            continue
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            result[key] += usage[key]
    if result['unknown_requests']:
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            result[key] = None
    return result


def savings(before, after):
    if before is None or after is None:
        return {'tokens': None, 'percent': None}
    return {'tokens': before - after, 'percent': round(100 * (before - after) / before, 2) if before else None}


PHASES = {'chat': 'Ручной чат', 'dialogue': 'Подготовка диалога', 'summary': 'Создание summary',
          'full': 'Случай 1 · без сжатия', 'compressed': 'Случай 2 · со сжатием'}


class TokenTracker:
    """Actual cost and input growth are separate measures. Store usage, not history."""

    def __init__(self):
        self.rows = []
        self.phases = {}
        self.previous_input = {}

    def record(self, trace, phase):
        row = {'usage': trace.get('usage')}
        self.rows.append(row)
        self.phases.setdefault(phase, []).append(row)
        current = usage_total([row])
        input_tokens = current['prompt_tokens']
        previous = self.previous_input.get(phase)
        delta = None if input_tokens is None or previous is None else input_tokens - previous
        self.previous_input[phase] = input_tokens
        return {'request_number': len(self.rows), 'phase': phase,
                'phase_request_number': len(self.phases[phase]),
                'prompt_tokens': input_tokens, 'input_delta': delta,
                'completion_tokens': current['completion_tokens'], 'request_tokens': current['total_tokens'],
                'phase_total': usage_total(self.phases[phase])['total_tokens'],
                'run_total': usage_total(self.rows)['total_tokens']}


def progress_rows(report):
    """Also reconstruct the same ledger for reports saved before live telemetry."""
    tracker = TokenTracker()
    stages = [('dialogue', report['dialogue']), ('summary', report['compressions']),
              ('full', report['probes']['full']), ('compressed', report['probes']['compressed'])]
    for phase, rows in stages:
        for row in rows:
            for trace in row['trace']:
                yield tracker.record(trace, phase)
