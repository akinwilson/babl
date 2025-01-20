from transformers import T5Tokenizer, T5ForConditionalGeneration
import os 
import torch 
from babl.model.T5.eval import clean 
from pprint import pprint 
from pathlib import Path 



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

    model_path= Path(os.getenv("MODEL_PATH", f"/usrs/src/app/outputs/{os.getenv('MODEL_NAME', 't5-small')}/checkpoint-1"))
    tok_path=  model_path.parent

    tok = T5Tokenizer.from_pretrained(tok_path)
    m = T5ForConditionalGeneration.from_pretrained(model_path)

    Predictor(tok, m, max_len =128)("How many countries are there in the United Kingdom?")


# print(inference())

