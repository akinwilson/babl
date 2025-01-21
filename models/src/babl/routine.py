import torch
import json
import logging
import os
from pathlib import Path
# import transformers 
# transformers.logging.set_verbosity_info()
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from .data import prepare_dataset
from argparse import ArgumentParser 
from .model.T5.config import ModelArguments, DataArguments
from .metrics import test 
from .data import T2TDataCollator
from .models import MODELS_CHOICES, MODELS

import logging 
logger = logging.getLogger(__name__)


'''
Using huggingface Trainer class to fit the models; I dont particularly like their Trainer class and prefer using the 
primitives provided by pytorch and pytorch-lightning as they provide fine-grained control over all aspects; model architecture, 
training, validation and testing regimes, exporting the models, hardware resource aquirements etc. Nethertheless, for this application demonstration, 
we will used their implementation of the training routine. 

if I get enough time, I would like to:
TODO: 
    transfer fitting routine over to pytorch-lightning implementation. 

    implement: 
        recurrent encoder-decoder  architecture
        then  attention-based recurrent encoder-decoder architecture

    and show how their performs in comparison to these attention based encoder-decoder transformers lags behind at the cost of an explosion in parameter count. 
'''

def routine(args, model, tokenizer):

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)
    )
    # we will load the arguments from a json file,
    # Check out https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.TrainingArguments
    # to see what other arguments the TrainingArgument accepts 
    args_dict = {
    # "num_cores": 6,
    "model_name_or_path": args.model_name_or_path, #   "t5-small",
    "input_dir":  str(args.input_dir),
    "output_dir": str(args.output_dir) ,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "tpu_num_cores": 8,
    "do_train": True,
    "do_eval": True,  
    "remove_unused_columns": False, # this caused me many issues with the collator. Moving over to pytorch lighnting to handling training routine
    "num_train_epochs": 5,
    }

    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"[routine.py]::{output_dir=}")
    with open( output_dir / "params.json", "w") as f:
        json.dump(args_dict, f)

    model_args, data_args, train_args = parser.parse_json_file(json_file= output_dir / "params.json")
    
    logger.debug(f"[routine.py]::{model_args=}")
    
    
    cache_dir = Path(args.root_dir) / model_args.cache_dir 
    cache_dir.mkdir(parents=True, exist_ok=True)

    ### THS SHOULD BE PART OF PRIOR STEP OF PIPELINE 
    prepare_dataset(args, data_args)
    logger.debug("[routine.py]::finished preparing dataset")
    if (os.path.exists(output_dir) and train_args.do_train and not train_args.overwrite_output_dir ):
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    logger.warning(f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}")
    # logger.info(f"Training/evaluation parameters:\n{train_args}")

    # Set seed
    set_seed(train_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = tokenizer.from_pretrained(model_args.model_name_or_path,cache_dir=cache_dir)
    model = model.from_pretrained(model_args.model_name_or_path,cache_dir=cache_dir)
    logger.debug(f"[routine.py]::{tokenizer=}")
    
    
    # Get datasets
    train_filepath = input_dir / data_args.proccessed_train_filename
    val_filepath = input_dir / data_args.proccessed_val_filename
    
    train_dataset = torch.load(train_filepath)
    valid_dataset = torch.load(val_filepath)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
    )

    # Training
    if train_args.do_train:
        loss = trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None # load from fs if avialable else download
        )
        logger.info(f"loss:{loss}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

    # Evaluation
    results = {}
    if train_args.do_eval and int(train_args.local_rank) in [-1, 0]:
        logger.info("Validation".center(100,"="))

        eval_output = trainer.evaluate()

        output_eval_file = output_dir / "eval_results.txt"

        with open(output_eval_file, "w") as f:
            logger.info("Results".center(100,'-'))
            for key in sorted(eval_output.keys()):
                logger.info(f" {key} = {str(eval_output[key])}")
                f.write(f"{key} = {str(eval_output[key])}\n")

        results.update(eval_output)

    return results








if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input-max-len', default=128)
    parser.add_argument('--model-name-or-path', default='t5')
    parser.add_argument('--output-max-len', default=32)

    # retrive model name from default value of parser above. 
    model_name = [a.default  for a in parser._actions if "model-name-or-path" in "".join(a.option_strings)][0]

    # where ever root is chosen it needs to be one level above /inputs where input data is expected to be store. 
    # /inputs are filled with ./pull_data.sh script 

    root = Path(__file__).parent.parent.parent 

    output_dir = root / "outputs" / model_name
    input_dir = root / "inputs"
    cache_dir = root / "cache"

    parser.add_argument('--output-dir', default=output_dir)
    parser.add_argument('--input-dir', default=input_dir)
    parser.add_argument("--root-dir", default=root)

    args = parser.parse_args()

    # how to start routine. select model flavours given architecture name
    tm = MODELS[args.model_name]
    # select the first model flavour in the list 
    full_model_name = MODELS_CHOICES[args.model_name_or_path][0]
    print(f"{full_model_name=}")


    args.model_name_or_path = full_model_name
    print(f"{args.model_name_or_path=}")

    tok = tm['tok'].from_pretrained(full_model_name, cache_dir=cache_dir)
    model = tm['model'].from_pretrained(full_model_name, cache_dir=cache_dir)
    routine(args, tok, model)
    test(args, tok, model)
    
    print("FINISED".center("=", 100))
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
    print(s)
