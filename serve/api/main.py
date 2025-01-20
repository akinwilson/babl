
from fastapi import FastAPI, Request
import os
from .predictor import Predictor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path 
from .schema import * 
from fastapi.logger import logger

# expecting os.envs: 
# MODEL_NAME 

app = FastAPI(
    title="Conversational AI: Question and Answering",
    description=f"Deep learning transformer-based pretrained networks fined-tuned to the task of question and answering. Serving model: {os.getenv('MODEL_NAME', 't5-small')}",
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
    root= Path(__file__).parent.parent.parent

    model_path= Path(os.getenv("MODEL_PATH", root / f"outputs/{os.getenv('MODEL_NAME', 't5-small')}/checkpoint-1"))
    tok_path=  model_path.parent

    tok = T5Tokenizer.from_pretrained(tok_path)
    m = T5ForConditionalGeneration.from_pretrained(model_path)
    
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


@app.get("/")
def read_root():
    return {"KeyModel" : "Value"}



