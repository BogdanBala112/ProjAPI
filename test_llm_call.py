from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_call import Ticket, classify_ticket


@pytest.mark.asyncio
async def test_empty_text_raises_error():
    with pytest.raises(ValueError):
        await classify_ticket("", AsyncMock())


@pytest.mark.asyncio
async def test_whitespace_text_raises_value_error():
    with pytest.raises(ValueError):
        await classify_ticket("   ", AsyncMock())


@pytest.mark.asyncio
async def test_good_response_returns_ticket():
    expected = Ticket(
        category="billing",
        priority="high",
        summary="Charged twice",
    )

    # Namespace is an object that turns the arguments given into attributes for the object assigned to it.
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=expected,
                    refusal=None,
                )
            )
        ]
    )
    mock_parse = AsyncMock(return_value=fake_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(parse=mock_parse),
        )
    )

    result = await classify_ticket("I was charged twice", fake_client)

    assert result == expected
    mock_parse.assert_awaited_once()
