import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from llm_call import ModelResponseError, Ticket, classify_ticket


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client = AsyncOpenAI(api_key=api_key, timeout=10.0)
    app.state.openai_client = client
    app.state.llm_semaphore = asyncio.Semaphore(3)

    try:
        yield
    finally:
        await client.close()


app = FastAPI(lifespan=lifespan)


def get_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai_client


def get_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_semaphore


OpenAIClient = Annotated[AsyncOpenAI, Depends(get_client)]
LLMSemaphore = Annotated[asyncio.Semaphore, Depends(get_semaphore)]


class ClassifyTicket(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=300,
        description="The text of the ticket to classify",
    )

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty!")
        return value


class ClassifyBatchRequest(BaseModel):
    tickets: list[str] = Field(min_length=1, max_length=50)


class BatchSuccess(BaseModel):
    text: str
    status: Literal["success"] = "success"
    result: Ticket


class BatchFailure(BaseModel):
    text: str
    status: Literal["error"] = "error"
    error: str


class ClassifyBatchResponse(BaseModel):
    results: list[BatchSuccess | BatchFailure]


async def classify_batch_item(
    text: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
) -> BatchSuccess | BatchFailure:
    try:
        validated = ClassifyTicket(text=text)
    except ValidationError as error:
        message = error.errors()[0]["msg"]
        return BatchFailure(text=text, error=message)

    try:
        async with semaphore:
            result = await classify_ticket(validated.text, client)
        return BatchSuccess(text=text, result=result)
    except APIError:
        return BatchFailure(text=text, error="The model service failed")
    except ModelResponseError as error:
        return BatchFailure(text=text, error=f"Invalid model response: {error}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/tickets/classify", response_model=Ticket)
async def classify_ticket_endpoint(
    ticket: ClassifyTicket,
    client: OpenAIClient,
    semaphore: LLMSemaphore,
):
    try:
        async with semaphore:
            return await classify_ticket(ticket.text, client)
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The model service failed",
        ) from error
    except ModelResponseError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid model response: {error}",
        ) from error


@app.post("/tickets/classify-batch", response_model=ClassifyBatchResponse)
async def classify_ticket_batch_endpoint(
    batch: ClassifyBatchRequest,
    client: OpenAIClient,
    semaphore: LLMSemaphore,
):
    results = await asyncio.gather(
        *(classify_batch_item(text, client, semaphore) for text in batch.tickets)
    )
    return ClassifyBatchResponse(results=results)
