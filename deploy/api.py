import logging
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from text2speech.infer import Infer

logger = logging.getLogger(__name__)  # TODO: add logger for the server


class PredictPayload(BaseModel):
    text: str = Query(None, min_length=3)


model_name = "microsoft/speecht5_tts"  # TODO: add config
app = FastAPI(title="AudioPaper", description="convert a text to audio", version="0.0.1")


def construct_response(f):
    @wraps(f)
    def wrapper(request: Request, *args, **kargs) -> Dict:
        results = f(request, *args, **kargs)
        results["timestamp"] = datetime.now().isoformat()
        results["method"] = request.method
        results["url"] = request.url._url
        return results

    return wrapper


@app.on_event("startup")
def load_artifacts():
    global model
    model = Infer(model_name=model_name)


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check"""
    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}
    return response


@app.get("/args", tags=["Args"])
@construct_response
def _args(request: Request, filter: str = None) -> dict:
    """Get the meta data of model components."""
    args = model.metadata
    data = args.get(filter, args)
    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": data}
    return response


@app.post("/predict", tags=["Prediction"])
async def _predict(request: Request, payload: PredictPayload):
    """Creat Audio out of the given text"""
    speech = model.tts(payload.text, speaker_id=7000)
    list_speech = speech.cpu().detach().numpy().tolist()

    response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {"speech": list_speech}}
    return response
