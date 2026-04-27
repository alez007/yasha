"""OpenAI-specific utilities for request/response processing."""


def flatten_message_content(messages: list[dict]) -> list[dict]:
    """
    Ensure all messages in a list have string 'content' if possible.
    OpenAI protocol allows 'content' to be a list of parts (e.g. for vision).

    Many loaders (like llama.cpp) use simple Jinja templates that strictly
    expect a string. We flatten the list into a single string ONLY if all
    parts are text parts, to avoid stripping multi-modal data (images/audio)
    from models that might support them.
    """
    flattened = []
    for msg in messages:
        processed_msg = msg.copy()
        content = msg.get("content")
        if isinstance(content, list):
            # Only flatten if EVERY part is a text part
            is_all_text = all(isinstance(part, dict) and part.get("type") == "text" for part in content)
            if is_all_text:
                text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                processed_msg["content"] = " ".join(text_parts)
        flattened.append(processed_msg)
    return flattened
