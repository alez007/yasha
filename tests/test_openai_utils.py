from modelship.utils.openai import flatten_message_content


def test_flatten_message_content_string():
    messages = [{"role": "user", "content": "hello"}]
    assert flatten_message_content(messages) == messages


def test_flatten_message_content_list_text_only():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    expected = [{"role": "user", "content": "hello world"}]
    assert flatten_message_content(messages) == expected


def test_flatten_message_content_mixed_list():
    # Should NOT flatten if images are present
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "..."}},
            ],
        }
    ]
    assert flatten_message_content(messages) == messages


def test_flatten_message_content_multiple_messages():
    messages = [
        {"role": "system", "content": "you are a bot"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        },
    ]
    expected = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "a b"},
    ]
    assert flatten_message_content(messages) == expected


def test_flatten_message_content_none():
    messages = [{"role": "assistant", "content": None}]
    assert flatten_message_content(messages) == messages
