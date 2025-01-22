import torch 
import logging 
import string
import re

logger = logging.getLogger(__name__)


from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


class CallbackCollection:
    def __init__(self, data_path, args):

        self.data_path = data_path
        self.args = args

    def __call__(self):
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        early_stopping = EarlyStopping(
            mode="min", monitor="val_loss", patience=self.args.es_patience
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.args.model_dir,
            save_top_k=2,
            save_last=True,
            mode="min",
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_ttr:.2f}-{val_ftr:.2f}",
        )

        callbacks = {
            "checkpoint": checkpoint_callback,
            "lr": lr_monitor,
            "es": early_stopping,
        }
        # callbacks = [checkpoint_callback, lr_monitor, early_stopping]
        return callbacks




def clean(x):
    return x.replace("<pad>", "").replace("</s>", "").strip().lower()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class Predictor:

    def __init__(self, tokenizer, model, max_len=64):
        self.tok = tokenizer
        self.m = model
        self.max_len=max_len


    def format_input(self, question, context):
        return f"question: {question} context: {context} </s>"


    def encode(self, input):
        # tokens =  tok.tokenize(input)
        encodings= self.tok.encode_plus(input, pad_to_max_length=True,truncation=True, max_length=self.max_len)
        logger.debug(f"{encodings=}")
        return encodings 

    def decode(self, output):
        return self.tok.decode(output)


    def generate(self, input_ids, attention_mask):
        ans_encoded = self.m.generate(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        logger.debug(f"{ans_encoded=}")
        return clean("".join([self.decode(x) for x in ans_encoded]))

    def inference(self, question="What is the pythagorean theorem?", context="there are 8 planets in the solar system."):
        return self.generate(**self.encode(self.format_input(question, context)))   

    def __call__(self, question, context=""):
        return self.inference(question, context)