from babl.architecture.T5.train import main  
from babl.architecture.T5.data import prepare_dataset 
from argparse import ArgumentParser
from pathlib import Path 
import warnings 

warnings.filterwarnings("ignore")




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--input-max-len', default=128)
    parser.add_argument('--output-dir', default="outputs")
    parser.add_argument('--input-dir', default="inputs")
    print(Path(__file__).parent.parent)
    parser.add_argument("--root-dir", default=Path(__file__).parent.parent)
    parser.add_argument('--model-name-or-path', default='t5-small')
    parser.add_argument('--output-max-len', default=32)
    args = parser.parse_args()
    main(args)