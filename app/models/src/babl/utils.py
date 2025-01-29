import torch 
import logging 
import string
import re
import logging 
import numpy as np
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import onnx 
import onnxruntime

logger = logging.getLogger(__name__)

class CallbackCollection:
    def __init__(self,args):
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
            filename="{epoch}-{val_loss:.2f}-{val_EM:.2f}-{val_F1:.2f}",
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


class Predictor(torch.nn.Module):

    def __init__(self, tokenizer, model, input_max_len=64):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model
        self.input_max_len=input_max_len


    def format_input(self, question, context):
        return f"question: {question} context: {context} </s>"


    def encode(self, input):
        # tokens =  tok.tokenize(input)
        encodings= self.tokenizer.encode_plus(input, pad_to_max_length=True,truncation=True, max_length=self.input_max_len)
        logger.debug(f"{encodings=}")
        return encodings 

    def decode(self, output):
        return self.tokenizer.decode(output)


    def generate(self, input_ids, attention_mask):
        ans_encoded = self.model.generate(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        logger.debug(f"{ans_encoded=}")
        return clean("".join([self.decode(x) for x in ans_encoded]))

    def inference(self, question="What is the pythagorean theorem?", context="there are 8 planets in the solar system."):
        return self.generate(**self.encode(self.format_input(question, context)))   

    # def __call__(self, question, context=""):
    #     return self.inference(question, context)
    
    def forward(self,question, context):
           return self.inference(question, context)
    




class OnnxExporter:
    def __init__(
        self, model, model_name, output_dir, op_set=17
    ):

        self.model_name = model_name
        self.model = model

        self.output_dir = output_dir
        self.op_set = op_set
        self.model.eval()
        
        assert (
            not self.model.training
        ), "Model not in inference mode before exporting to onnx format"
        # Input to the model
        
        batch_size = 1
        # shape = (batch_size,) + input_shape
        # Get expected input dims from config cfg.processing_output_shape = (40, 241)
        # self.x_in = torch.randn(shape, device="cpu")
        self.QndC = {"question" :"What year was I born?", "context": "I was born 1995"}
        
        logger.info(f"Input for model tracing: {self.QndC}")
        self.x_out = self.model(**self.QndC)
        print(f"{self.x_out=}")
        # logger.info(f"Output given input for model tracing: {self.x_out.shape}")
        self.onnx_model_path = self.output_dir + "/model.onnx"

    def verify(self):
        logger.info("Verifying model has been saved correctly ... ")
        logger.info(f"Loading onnx model from path {self.onnx_model_path} ... ")
        model = onnx.load(self.onnx_model_path)
        onnx.checker.check_model(model)

    # def to_numpy(self, tensor):
    #     return (
    #         tensor.detach().cpu().numpy()
    #         if tensor.requires_grad
    #         else tensor.cpu().numpy()
    #     )

    # Export the model
    def __call__(self):
        # print("self.output_dir", self.output_dir)
        # print("self.output_path", output_path)
        # self.onnx_model_path = output_path
        logger.info(f"Onnx model output path: {self.onnx_model_path}")
        model = self.model
        positional_inputs = ("question: how old are you",'context: I was born 1995')

        torch.onnx.export(
            model=model,  # model being run
            args=positional_inputs,  # model input (or a tuple for multiple inputs)
            f=self.onnx_model_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=self.op_set,  # Only certain operations are available, 17 includes FFTs and IFFTs
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["question", "context",],  # the model's input names
            output_names=["answer"],  # the model's output names
            dynamic_axes={
                "question with context": {0: "batch_size"},  # variable length axes
                "answer": {0: "batch_size"},
            },
        )
        self.verify()
        self.inference_session()
        return self

    def inference_session(self):
        ort_session = onnxruntime.InferenceSession(
            str(self.onnx_model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # compute ONNX Runtime output prediction
        # logger.info(f"ort_session: {ort_session.__dict__}")
        # logger.info(f"ort_session.get_inputs(): {ort_session.get_inputs()}")
        ort_inputs = {ort_session.get_inputs()[0].name: list(self.QndC.values())}

        # logger.info(f"ort_inputs {ort_inputs}")
        ort_outs = ort_session.run(None, ort_inputs)
        print(f"{ort_outs=}")
        # compare ONNX Runtime and PyTorch results

        np.testing.assert_allclose(
            self.to_numpy(self.x_out), ort_outs[0], rtol=1e-03, atol=1e-05
        )
        logger.info(
            "Exported model has been tested with ONNXRuntime, and the result looks good!"
        )
