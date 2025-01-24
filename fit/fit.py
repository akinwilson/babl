import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from dataclasses import dataclass
from babl.data import TextDataModule
from babl.utils import CallbackCollection
from babl.routine import Routine
from babl.models import MODELS_CHOICES, MODELS
from babl.config import T5 as T5Config
from babl.config import Data as DataArgs
from babl.utils import Predictor 
import torch 
from argparse_dataclass import ArgumentParser
import warnings 
warnings.filterwarnings("ignore")
import logging 

logging.basicConfig(
    filename=  Path(__file__).parent / 'fit.log', encoding='utf-8',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
    # gging.INFO, # if train_args.local_rank in [-1, 0] else logging.WARN, # for distributed training 
)



WHITE='\033[1:97m]'
GREEN='\033[0;32m'
LIGHTPURPLE='\033[1;35m'
LIGHTCYAN='\033[1;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'
s = """
                                                                                
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
                                                                                
                                                                                
"""



class Fitter:
    def __init__(
        self,
        model,
        tokenizer,
        model_name,
        data_args,
        mini_dataset = True, 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.args = data_args
        self.mini_dataset = mini_dataset
        self.data_module = TextDataModule(data_args=data_args, tokenizer=tokenizer, mini_dataset=mini_dataset) 
        self.trainer = None
        ####################################################################################
        @dataclass
        class FittingArgs:
            es_patience: int = 2
            model_dir = Path("/home/nameduser/Code/babl/outputs") / model_name
            max_epoch: int = 5
            fast_dev_run: bool = False
            mini_dataset: bool = True 
            def __post_init__(self):
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self.model_dir  =  str(self.model_dir)

        ####################################################################################

        self.args = FittingArgs()

    def setup(self):
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()

        return train_loader, val_loader, test_loader

    def callbacks(self):
        # cfg_fitting = self.cfg_fitting
        callback_collection = CallbackCollection(self.args)
        return callback_collection()

    def __call__(self):

        logger = TensorBoardLogger(
            save_dir=self.args.model_dir,
            name="lightning_logs",
        )
        # Model = self.model
        # get loaders and datamodule to access input shape
        train_loader, val_loader, test_loader = self.setup()
        print("Created training, validating and test loaders .... ")
        # get input shape for onnx exporting
        # input_shape = data_module.input_shape
        # init model
        # kwargs = {}
        # model = Model(**kwargs)

        # setup training, validating and testing routines for the model
        routine = Routine(self.model)

        # Init a trainer to execute routine
        callback_dict = self.callbacks()
        callback_list = [v for (_, v) in callback_dict.items()]
        number_devices = os.getenv("CUDA_VISIBLE_DEVICES", "1,").split(",")
        try:
            number_devices.remove("")
        except ValueError:
            pass

        self.trainer = Trainer(
            accelerator="cpu",
            devices=len(number_devices),
            # strategy=os.getenv("STRATEGY", "ddp_notebook"),
            sync_batchnorm=True,
            logger=logger,
            max_epochs=self.args.max_epoch,
            callbacks=callback_list,
            num_sanity_val_steps=2,
            # resume_from_checkpoint=self.cfg_fitting.resume_from_checkpoint,
            gradient_clip_val=1.0,
            fast_dev_run=self.args.fast_dev_run,
        )

        self.trainer.fit(
            routine, train_dataloaders=train_loader, val_dataloaders=val_loader
        )  # ,ckpt_path=PATH)

        if self.args.fast_dev_run:
            # issue with finding best weights path for in fast dev run using last model weights
            model_ckpt_path = callback_dict["checkpoint"].__dict__["last_model_path"]
        else:
            model_ckpt_path = callback_dict["checkpoint"].__dict__["best_model_path"]

        self.trainer.test(
            dataloaders=test_loader,
            ckpt_path=model_ckpt_path,
        )
        print(WHITE + s + RESET)
        # Return the input_shapes and trainer of the model for exporting
        return self
    

if __name__=="__main__":

    import pickle
    # from babl.config import Fitting as FittingArgs
    from babl.config import Args
    import argparse


    # Args = FittingArgs

    from pprint import pprint

    parser = ArgumentParser(Args,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    args = parser.parse_args()
    pprint(args)


    # model_name = "t5"
    # full_model_name = MODELS_CHOICES[model_name][0]
    # t_w_m = MODELS[model_name]

    # t = t_w_m["tok"]
    # m = t_w_m["model"]

    # tokenizer = t.from_pretrained(full_model_name)
    # model = m.from_pretrained(full_model_name, **T5Config().__dict__)
    # model.train()
    # # overwritting the MODEL_NAME with the full version
    # os.environ['MODEL_NAME'] = full_model_name
    # fitter= Fitter(model=model, model_name=full_model_name, tokenizer=tokenizer, data_args=args)()

    # # os.environ['MODEL_NAME']
    # # during distributed training accessing the model is further down the module tree
    # if torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     model = fitter.trainer.model.model
    #     # fitter.trainer.model
    #     tokenizer = fitter.data_module.tokenizer



    # # args = DataArgs()
    # predictor = Predictor(tokenizer=tokenizer, model=model, input_max_len=args.input_max_len)

    # # args = FittingArgs()
    # with open(Path(args.model_dir) / "model.pickle", "wb") as f:
    #     pickle.dump(predictor, f)
    # with open(Path(args.model_dir) / "model.pickle", "rb") as f:
    #     loaded_model = pickle.load(f)

    # Q = LIGHTPURPLE + "Question: What is the moon?" + RESET
    # C = LIGHTCYAN + "Context: the solar system contains planets, moons, meteors" + RESET

    # print(f"Reloaded saved pickled model\n{Q}\n{C}\nResponse:\n\n{GREEN}{loaded_model(Q, C)}{RESET}")