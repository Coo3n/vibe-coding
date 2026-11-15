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
