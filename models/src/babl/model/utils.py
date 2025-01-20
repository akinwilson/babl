
def clean(x):
    return x.replace("<pad>", "").replace("</s>", "").strip().lower()


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