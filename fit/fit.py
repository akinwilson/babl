from babl.metrics import test  
from babl.routine import routine
from babl.models import MODELS, MODELS_CHOICES 
from argparse import ArgumentParser
from pathlib import Path 
import warnings 
import os 
warnings.filterwarnings("ignore")

import logging 
# Setup logging
logging.basicConfig(
    filename=  Path(__file__).parent / 'fit.log', encoding='utf-8',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO, # if train_args.local_rank in [-1, 0] else logging.WARN, # for distributed training 
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = ArgumentParser()
    ## can select models via 
    # t5,llama,bloom,bert
    parser.add_argument('--input-max-len', default=128)
    parser.add_argument('--model-name-or-path', default=os.getenv("MODEL_NAME", "t5"), choices=MODELS_CHOICES[os.getenv("MODEL_NAME", "t5")])
    parser.add_argument('--output-max-len', default=32)
    # retrive model name from default value of parser above. 
    

    model_name = [a.default  for a in parser._actions if "model-name-or-path" in "".join(a.option_strings)][0]

    root = Path(__file__).parent.parent
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = root / "outputs" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = root / "inputs"
    parser.add_argument('--output-dir', default=output_dir)
    parser.add_argument('--input-dir', default=input_dir)
    parser.add_argument("--root-dir", default=root)

    args = parser.parse_args()
 

    tm = MODELS[model_name]
    full_model_name = MODELS_CHOICES[model_name][0]
    args.model_name_or_path=full_model_name

    logger.debug(f"[fit.py]::__main__{full_model_name=}")
    logger.debug(f"[fit.py]::__main__{args.model_name_or_path=}")
    logger.warning('Watch out!')

    tok = tm['tok'].from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    model = tm['model'].from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    
    routine(args, model=model, tokenizer=tok)
    test(args, model=model, tokenizer=tok)

    logger.info("FINISHED".center(100, "="))


    # see : https://en.wikipedia.org/wiki/ANSI_escape_code
    grey='\033[1;90m'
# reset 
    reset='\033[0m'
    s = '''
                            @@@@@@@@@@@@@@@@@@@@@@@@@                           
                      ,@@@@@@@@@@@            #@@@@@@@@@@@                      
                   @@@@@@      @@@@           @@@      @@@@@@&                  
                @@@@@          &@@@   ,@@@@@@@@@@          @@@@@.               
              @@@@@@      @@@@@@@@@   ,@@@   /@@   @@@      @@@@@@@             
            @@@@  @@@@@    @@@@       ,@@@        @@@@@@@@@@@@# @@@@@           
          @@@@      @@@@@   @@@@  .@@@@@@@@@@@  @@@@   @@@@@      @@@@          
         @@@@     @@@@@       @@@@@@@@@@@@@@@@@@@@@                 @@@@        
        @@@@     @@@@@.    @@@@@@@@   ,@@@   @@@@@@@@@    @@@@@@     @@@@       
       @@@@         @@@@@@@@@    @@@@@@@@@@@@@@@   @@@@@@@@@@@@@@     @@@(      
      @@@@@@@@@@@@(     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@      
      @@@.    @@@@     @@@  %@@@@@@@@@@@@@@@@@@@@@@@  @@@@     @@      @@@&     
      @@@     @@@     @@@@  @@@@@@@@@@@@@@@@@@@@@@@@   @@@             @@@@     
      @@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@     
      @@@             @@@@  @@@@@@@@@@@@@@@@@@@@@@@@   @@@     @@@#    @@@@     
      @@@(    @@@@     @@@  @@@@@@@@@@@@@@@@@@@@@@@@@ @@@@     @@@.    @@@#     
      @@@@@@@@@@@@@    %@@@@@@.@@@@@@@@@@@@@@@@@@@@@@@@@@     &@@@@@@@@@@@      
       @@@@     @@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@   @@@@@@@@@@.        @@@       
        @@@@     @@@@@     @@@@@@@@    @@@    @@@@@@@      @@@@.     @@@#       
         @@@@                %@@@@@@@@@@@@@@@@@@@@@      %@@@@     /@@@         
          @@@@@     @@@@@@  @@@@    @@@@@@@@%   #@@@#  *@@@@.     @@@@          
            @@@@, @@@@/@@@@@@@@        @@@        @@@@    @@@@@ @@@@            
              @@@@@@       @@   @@@@%  @@@  .@@@@@@@@@      /@@@@@,             
                @@@@@@          @@@@@@@@@@   @@@@          @@@@@                
                   @@@@@@@     @@@@           @@@(     @@@@@@                   
                       @@@@@@@@@@@            .@@@@@@@@@@                       
                            .@@@@@@@@@@@@@@@@@@@@@@@                            
        '''
    print( grey + s + reset)
