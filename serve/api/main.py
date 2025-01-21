
from fastapi import FastAPI, Request
import os
from .predictor import Predictor
from babl.models import MODELS, MODELS_CHOICES

# from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path 
from .schema import * 
from fastapi.logger import logger

# expecting os.envs: 
# MODEL_NAME 

app = FastAPI(
    title="Conversational AI: Question and Answering",
    description=f"Deep learning transformer-based pretrained networks fined-tuned to the task of question and answering. Serving model: {os.getenv('MODEL_NAME', 't5')}",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
)

@app.on_event("startup")
async def startup_event():
    """
    Load model into memory ready for endpoint to receive requests 
    """

    # root= Path(__file__).parent.parent.parent
    # model_path= Path(os.getenv("MODEL_PATH", root / f"outputs/{os.getenv('MODEL_NAME', 't5')}/checkpoint-1"))
    # tok_path=  model_path.parent

    model_path= [p for p in  list(Path("/usr/src/app/outputs").iterdir()) if os.getenv('MODEL_NAME', 't5') in str(p) ][0] / 'checkpoint-1'
    tok_path=  model_path.parent
    # model_path= Path(os.getenv("MODEL_PATH", f"/usr/src/app/outputs/{os.getenv('MODEL_NAME', 't5-small')}/checkpoint-1"))
    tm = MODELS[os.getenv('MODEL_NAME', 't5')]
    full_model_name = MODELS_CHOICES[os.getenv('MODEL_NAME', 't5')][0]
 
    tok = tm['tok'].from_pretrained(tok_path)
    m = tm['model'].from_pretrained(model_path)
    app.package = {"model": Predictor(tok, m, max_len =128)}
    # ("How many countries are there in the United Kingdom?")



@app.post("/api/v1/predict",
          response_model= InferenceResult ,
          responses={ 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse} 
                     })
def predict(request: Request, body: InferenceInput):
    '''
    Response to question 
    '''
    logger.info(f"Received: {body}")

    result = app.package['model'](question=body.question, context=body.context)
    return {"answer": result}


# @app.get("/")
# def read_root():
#     return {"KeyModel" : "Value"}



