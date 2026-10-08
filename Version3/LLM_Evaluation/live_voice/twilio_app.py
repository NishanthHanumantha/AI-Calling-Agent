"""Eval-only FastAPI webhooks. Bind to 127.0.0.1:8001. Production /voice is never used."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .bootstrap import bootstrap
from .handlers import handle_recording, handle_speech, start_call
from .runtime import get_runtime

bootstrap()

app = FastAPI(title="LLM Evaluation Live Voice", docs_url=None, redoc_url=None, openapi_url=None)


def xml(content: str, status_code: int = 200) -> Response:
    return Response(content=content, media_type="application/xml", status_code=status_code)


def _query(request: Request) -> dict[str, str]:
    return {str(key): str(value) for key, value in request.query_params.items()}


@app.api_route("/eval/voice", methods=["GET", "POST"])
async def eval_voice(request: Request):
    form: dict = {}
    if request.method == "POST":
        form = dict(await request.form())
    elif request.query_params.get("CallSid"):
        form = dict(request.query_params)
    return xml(start_call(_query(request), form))


@app.post("/eval/handle-speech")
async def eval_handle_speech(request: Request):
    form = dict(await request.form())
    return xml(await handle_speech(form))


@app.post("/eval/recording")
async def eval_recording(request: Request):
    form = dict(await request.form())
    return JSONResponse(handle_recording(form))


@app.get("/eval/status")
async def eval_status():
    runtime = get_runtime()
    return {
        "status": "ok",
        "bind": "127.0.0.1",
        "sessions": len(runtime.store.by_call_sid),
        "pending": len(runtime.store.pending),
    }
