#!/usr/bin/env python 
from dataclasses import dataclass, field
from pathlib import Path 
import os 
from argparse_dataclass import ArgumentParser




@dataclass
class T5:
    vocab_size : int = field(default=32128, metadata={"help": "Size of vocabulary of transformer."})
    d_model :int = 512
    d_kv :int = 64
    d_ff :int = 2048
    num_layers :int = 6
    num_decoder_layers = None
    num_heads :int = 8
    relative_attention_num_buckets : int = 32
    relative_attention_max_distance : int = 128
    dropout_rate : float = 0.1
    layer_norm_epsilon : float = 1e-06
    initializer_factor: float = 1.0
    feed_forward_proj :str = 'relu'
    is_encoder_decoder: bool = True
    use_cache : bool = True
    pad_token_id :int  = 0
    eos_token_id : int = 1
    classifier_dropout :int = 0.0


os.environ['MODEL_NAME'] = 't5'


@dataclass
class Data:
    input_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for input sequence"})
    output_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for output sequence"})

#
#####################################################################################
@dataclass
class Fitting(Data):
    es_patience: int = 2
    model_dir : Path = Path("/home/nameduser/Code/babl/outputs") / os.environ['MODEL_NAME']
    max_epoch: int = 5
    fast_dev_run: bool = False
    mini_dataset: bool = True ,  field(
        metadata={
            "help": "Whether to use a baby dataset of 128 to test the fitting routine"
        }
    )

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
#         self.model_dir  =  str(self.model_dir)
####################################################################################
@dataclass
class Data:
    input_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for input sequence"})
    output_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for output sequence"})


from dataclasses import dataclass, field




@dataclass
class B:
    data_value2 : float = field(default=0.9, metadata={'help': 'info about data_value2'})

    def __post_init__(self):
        self.data_value2 += 10 

@dataclass
class A(B):
    data_value1 : str = field(default='foo', metadata={'help': 'info about data_value2'})

@dataclass
class B:
    data_value2 : float = field(default=0.9, metadata={'help': 'info about data_value2'})
    def __post_init__(self):
        self.data_value2 += 10 


C = A()



print(C.data_value2)




# testing merging class 


if __name__ == "__main__":
    from pprint import pprint
    parser = ArgumentParser(Fitting)
    args = parser.parse_args()
    pprint(args)














