from babl.model.T5.train import main
from babl.model.T5.eval import test  
from babl.model.T5.data import prepare_dataset 
from argparse import ArgumentParser
from pathlib import Path 
import warnings 

warnings.filterwarnings("ignore")




if __name__ == "__main__":

    parser = ArgumentParser()

 

    parser.add_argument('--input-max-len', default=128)
    parser.add_argument('--model-name-or-path', default='t5-small')
    parser.add_argument('--output-max-len', default=32)
    # retrive model name from default value of parser above. 
    model_name = [a.default  for a in parser._actions if "model-name-or-path" in "".join(a.option_strings)][0]

    root = Path(__file__).parent.parent
    output_dir = root / "outputs" / model_name
    input_dir = root / "inputs"
    parser.add_argument('--output-dir', default=output_dir)
    parser.add_argument('--input-dir', default=input_dir)
    parser.add_argument("--root-dir", default=root)

    args = parser.parse_args()
    main(args)
    test(args)