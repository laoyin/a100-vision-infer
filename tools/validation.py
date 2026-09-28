"""CPU-only request validation before native IPC admission."""
import math


def validate_body(body, max_context):
    if not isinstance(body, dict):
        raise ValueError('Request must be a JSON object')
    limits = {'temperature': (0, 100), 'top_p': (0, 1),
              'repetition_penalty': (0, 100), 'timeout_seconds': (0, 86400)}
    for key, (low, high) in limits.items():
        if key not in body:
            continue
        value = body[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid {key}')
        if key in ('top_p', 'repetition_penalty') and value == 0:
            raise ValueError(f'{key} must be positive')
    for key, low, high in [('max_tokens', 1, max_context-1), ('top_k', 0, 2147483647), ('seed', 0, 2**64-1)]:
        if key in body and (type(body[key]) is not int or not low <= body[key] <= high):
            raise ValueError(f'Invalid {key}')
    for key in ('stream', 'enable_thinking'):
        if key in body and type(body[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise ValueError('messages must be a nonempty list of objects')
    for message in messages:
        content = message.get('content')
        if isinstance(content, list) and not all(isinstance(p, dict) for p in content):
            raise ValueError('Content parts must be objects')
