# Ticket Classifier API

An asynchronous FastAPI service that uses an OpenAI model to classify support
tickets by category and priority. It supports individual and batch requests,
validates structured model output with Pydantic, and limits concurrent model
calls with a shared semaphore.

## Features

- Single-ticket and batch classification
- Structured, validated LLM responses
- Asynchronous OpenAI requests
- Application-wide concurrency limit of three model calls
- Per-item errors in batch requests
- FastAPI dependency injection and lifespan resource management
- Docker image with a health check
- Mocked unit and API tests

## Classification schema

Each successful classification contains:

- `category`: `billing`, `technical`, `account`, `feedback`, or `other`
- `priority`: `low`, `medium`, `high`, or `urgent`
- `summary`: a non-empty summary of up to 300 characters

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- Docker (optional)

## Local setup

Install the locked dependencies:

```bash
uv sync
```

Create a local `.env` file:

```env
OPENAI_API_KEY=your-api-key
```

Never commit `.env` or put API keys in the Docker image.

Start the development server:

```bash
uv run fastapi dev main.py
```

Open the interactive API documentation:

```text
http://localhost:8000/docs
```

## API endpoints

### Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Classify one ticket

```http
POST /tickets/classify
Content-Type: application/json
```

Request:

```json
{
  "text": "I was charged twice for my subscription."
}
```

Example response:

```json
{
  "category": "billing",
  "priority": "high",
  "summary": "The customer was charged twice for their subscription."
}
```

### Classify a batch

```http
POST /tickets/classify-batch
Content-Type: application/json
```

Request:

```json
{
  "tickets": [
    "I was charged twice.",
    "I cannot log into my account.",
    "   "
  ]
}
```

The batch accepts between 1 and 50 tickets. Results remain in input order, and
one invalid ticket does not prevent other tickets from being classified.

Example response:

```json
{
  "results": [
    {
      "text": "I was charged twice.",
      "status": "success",
      "result": {
        "category": "billing",
        "priority": "high",
        "summary": "The customer was charged twice."
      }
    },
    {
      "text": "I cannot log into my account.",
      "status": "success",
      "result": {
        "category": "account",
        "priority": "high",
        "summary": "The customer cannot access their account."
      }
    },
    {
      "text": "   ",
      "status": "error",
      "error": "Value error, Text must not be empty!"
    }
  ]
}
```

## Tests

Tests mock the OpenAI client, so they do not require an API key or make paid
network requests.

```bash
uv run pytest
```

## Docker

Build the image:

```bash
docker build -t ticket-classifier .
```

Run the container:

```bash
docker run -d \
  --name ticket-classifier-api \
  -p 8000:8000 \
  --env-file .env \
  ticket-classifier
```

Check its status and logs:

```bash
docker ps
docker logs -f ticket-classifier-api
```

Stop and remove it:

```bash
docker rm -f ticket-classifier-api
```

## Project structure

```text
.
├── main.py            # FastAPI app, dependencies, endpoints, and lifespan
├── llm_call.py        # Ticket models and OpenAI classification logic
├── test_api.py        # API tests
├── test_llm_call.py   # Classification unit tests
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

## Current architecture

The application creates one reusable asynchronous OpenAI client and one shared
semaphore during startup. FastAPI injects them into endpoints using
dependencies. The semaphore allows at most three concurrent model calls,
including calls submitted through different HTTP requests.
