"""Explicit Qwen multimodal string formatting, without a vLLM dependency."""


def vllm_string_messages(messages, image_marker):
    """Match vLLM's default non-interleaved string mode for text/image parts.

    Images precede newline-joined text. Explicit user-written placeholders are
    rejected instead of silently duplicating or reordering them.
    """
    result = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            result.append(dict(message))
            continue
        images, texts = [], []
        for part in content:
            if part['type'] == 'image':
                images.append(image_marker)
            elif part['type'] == 'text':
                if image_marker in part['text']:
                    raise ValueError('Explicit image placeholders require separate formatting validation')
                texts.append(part['text'])
            else:
                raise ValueError('String compatibility supports text and still images only')
        text = '\n'.join(texts)
        result.append(dict(message, content='\n'.join(images+([text] if text else []))))
    return result
