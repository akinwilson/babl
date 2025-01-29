# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import logging 
import os 
import torch 
# from babl.model.T5.eval import clean 
from babl.utils import clean
from babl.models import MODELS, MODELS_CHOICES
from pprint import pprint 
from pathlib import Path 


logger = logging.getLogger(__name__)

# MODELS_CHOICES = {
#     "t5": ['t5-small', 't5-base', 't5-large','t5-3b','t5-11b'],
#     "llama": ['meta-llama/Llama-3.3-70B-Instruct'],
#     "bert": ['google-bert/bert-base-uncased'],
#     "bloom": ["bigscience/bloom"]}
# # just choosing smallest t5 model for now 
# MODELS = { 
#     "t5": {"tok": T5Tokenizer, "model": T5ForConditionalGeneration},
#     "llama":{"tok": AutoTokenizer, "model":AutoModelForCausalLM} ,
#     "bert": {"tok":AutoTokenizer, "model":AutoModelForMaskedLM},
#     "bloom": {"tok":AutoTokenizer, "model":AutoModelForCausalLM}}



class Predictor:

    def __init__(self, tokenizer, model, max_len=32):
        self.tok = tokenizer
        self.m = model
        self.max_len=max_len


    def format_input(self, question, context):
        return f"question: {question} context: {context} </s>"


    def encode(self, input):
        # tokens =  tok.tokenize(input)
        encodings= self.tok.encode_plus(input, pad_to_max_length=True,truncation=True, max_length=self.max_len)
        # pprint(f"{encodings=}")
        return encodings 

    def decode(self, output):
        return self.tok.decode(output)


    def generate(self, input_ids, attention_mask):
        ans_encoded = self.m.generate(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        # print(f"{ans_encoded=}")
        return clean("".join([self.decode(x) for x in ans_encoded]))

    def inference(self, question="What is the pythagorean theorem?", context="there are 8 planets in the solar system."):
        return self.generate(**self.encode(self.format_input(question, context)))   

    def __call__(self, question, context=""):
        return self.inference(question, context)



if __name__ == "__main__":

    # NOTICE: we assume that the folder containing the models is called 'checkpoint-1'
    #         if more than one checkpoint is available, can choose the checkpoint with the highest number 
    #         or only ever let one checkpoint folder be present; `checkpoint-1`


    model_path= [p for p in  list(Path("/usr/src/app/outputs").iterdir()) if os.getenv('MODEL_NAME', 't5') in str(p) ][0] / 'checkpoint-1'
    tok_path=  model_path.parent
    # model_path= Path(os.getenv("MODEL_PATH", f"/usr/src/app/outputs/{os.getenv('MODEL_NAME', 't5-small')}/checkpoint-1"))
    tm = MODELS[os.getenv('MODEL_NAME', 't5')]
    full_model_name = MODELS_CHOICES[os.getenv('MODEL_NAME', 't5')][0]
 
    tok = tm['tok'].from_pretrained(tok_path)
    m = tm['model'].from_pretrained(model_path)

    # tok = T5Tokenizer.from_pretrained(tok_path)
    # m = T5ForConditionalGeneration.from_pretrained(model_path)

    Predictor(tok, m, max_len =128)("How many countries are there in the United Kingdom?")


# print(inference())

