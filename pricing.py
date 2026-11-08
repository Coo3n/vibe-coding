"""Сценарная стоимость по публичным тарифам, не фактическое списание."""

PRICING = {
    'currency': 'USD',
    'status': 'public_price_estimate',
    'checked_on': '2026-11-08',
    'formula': '(prompt_tokens * input_per_million + completion_tokens * output_per_million) / 1000000',
    'assumptions': 'Весь вход без кэша; без batch-скидок, налогов и дополнительных сборов. '
                   'Reasoning повторно не прибавляется к completion_tokens. '
                   'Это расчётный сценарий по публичным сервисам, не фактическое списание.',
    'rates': {
        'llm-small': {
            'public_model': 'qwen3.8-27b', 'input_per_million': 0.50, 'output_per_million': 3.00,
            'source': 'https://www.qwencloud.com/models/qwen3.8-27b',
            'note': 'Тариф размещённой Qwen 27B используется как ориентир для FP8-версии; '
                    'точная идентичность конфигураций не подтверждена.',
        },
        'llm-medium': {
            'public_model': 'DeepSeek-V4-Flash', 'input_per_million': 0.22, 'output_per_million': 0.66,
            'source': 'https://api-docs.deepseek.com/news/news260813',
            'price_table': 'https://api-docs.deepseek.com/img/v4_260813_price_en.png',
            'note': 'Исторический тариф с 16.08.2026, off-peak, вход без кэша. '
                    'Сценарный ориентир для checkpoint 0731, не текущая котировка этого checkpoint. '
                    'В peak тариф вдвое выше: $0.44/$1.32 за 1M токенов.',
        },
        'coder': {
            'public_model': 'glm-5.3-flash', 'input_per_million': 0.15, 'output_per_million': 0.50,
            'source': 'https://docs.z.ai/guides/overview/pricing',
            'note': 'Публичный тариф GLM-5.3-Flash используется как ориентир для ID glm-53-flash; '
                    'соответствие внутреннего ID публичной модели принято как допущение.',
        },
    },
}


def estimate_cost(route, prompt_tokens, completion_tokens):
    rate = PRICING['rates'].get(route)
    if rate is None or any(type(v) is not int or v < 0 for v in (prompt_tokens, completion_tokens)):
        return None
    return (prompt_tokens * rate['input_per_million']
            + completion_tokens * rate['output_per_million']) / 1_000_000
