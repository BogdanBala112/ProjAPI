from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    category: Literal["billing", "technical", "account", "feedback", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(
        min_length=1,
        max_length=300,
        description="A summary regarding the issue in the ticket",
    )


class ModelResponseError(RuntimeError):
    """Raised when the model does not return a usable structured response."""


def get_system_prompt() -> str:
    return """
    You are a helpful assistant that classifies tickets into categories and priorities.
    """


async def classify_ticket(text: str, client: AsyncOpenAI) -> Ticket:
    if not text.strip():
        raise ValueError("Text must not be empty!")

    response = await client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": text},
        ],
        response_format=Ticket,
    )

    message = response.choices[0].message
    if message.parsed is None:
        if message.refusal:
            raise ModelResponseError(f"Model refused the request: {message.refusal}")
        raise ModelResponseError("The model did not return a parsed ticket")

    return message.parsed
