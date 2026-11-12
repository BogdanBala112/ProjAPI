import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from llm_call import Ticket
from main import app, get_client, get_semaphore


def parsed_response(ticket: Ticket):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=ticket, refusal=None)
            )
        ]
    )


def test_batch_preserves_order_and_contains_item_errors():
    billing = Ticket(
        category="billing",
        priority="high",
        summary="Charged twice",
    )
    account = Ticket(
        category="account",
        priority="high",
        summary="Cannot log in",
    )
    mock_parse = AsyncMock(
        side_effect=[
            parsed_response(billing),
            parsed_response(account),
        ]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(parse=mock_parse),
        )
    )

    app.dependency_overrides[get_client] = lambda: fake_client
    app.dependency_overrides[get_semaphore] = lambda: asyncio.Semaphore(3)

    try:
        response = TestClient(app).post(
            "/tickets/classify-batch",
            json={
                "tickets": [
                    "I was charged twice",
                    "   ",
                    "I cannot log in",
                ]
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["status"] for result in results] == [
        "success",
        "error",
        "success",
    ]
    assert results[0]["result"]["category"] == "billing"
    assert results[1]["text"] == "   "
    assert results[2]["result"]["category"] == "account"
    assert mock_parse.await_count == 2
