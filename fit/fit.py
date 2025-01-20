from babl.model.T5.train import main
from babl.model.T5.eval import test  
from babl.model.T5.data import prepare_dataset 

# from babl.model.llama import tokenizer as llama_tok 
# from babl.model.llama import model as llama_model 
# from babl.model.bert import tokenizer as bert_tok
# from babl.model.bert import  model as bert_model
# from babl.model.bloom import  model as bloom_model
# from babl.model.bloom import tokenizer as bloom_tok

from argparse import ArgumentParser
from pathlib import Path 
import warnings 
import os 

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--input-max-len', default=128)
    parser.add_argument('--model-name-or-path', default=os.getenv("MODEL_NAME", "t5"), choices=MODELS_CHOICES[os.getenv("MODEL_NAME", "t5")])
    parser.add_argument('--output-max-len', default=32)
    # retrive model name from default value of parser above. 
    model_name = [a.default  for a in parser._actions if "model-name-or-path" in "".join(a.option_strings)][0]

    root = Path(__file__).parent.parent
    output_dir = root / "outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = root / "inputs"
    parser.add_argument('--output-dir', default=output_dir)
    parser.add_argument('--input-dir', default=input_dir)
    parser.add_argument("--root-dir", default=root)

    args = parser.parse_args()
    main(args)
    test(args)