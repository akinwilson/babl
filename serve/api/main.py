from typing import Union 
from fastapi import FastAPI
import os

os.getenv("")

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


@app.get("/")
def read_root():
    return {"KeyModel" : "Value"}

