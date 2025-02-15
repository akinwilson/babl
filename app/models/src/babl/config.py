from dataclasses import dataclass, field
from pathlib import Path 
import os 
from babl.models import MODELS_CHOICES



@dataclass
class T5:
    vocab_size : int = field(default=32128, metadata={"help": f"The vocabulary size of {__name__} transformer."})
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

@dataclass
class Bloom:
    vocab_size = 250880
    hidden_size = 64
    n_layer = 2
    n_head = 8
    layer_norm_epsilon = 1e-05
    initializer_range = 0.02
    use_cache = True
    bos_token_id = 1
    eos_token_id = 2
    apply_residual_connection_post_layernorm = False
    hidden_dropout = 0.0
    attention_dropout = 0.0
    pretraining_tp = 1
    slow_but_exact = False

@dataclass
class Bert:
    vocab_size = 30522
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 3072
    hidden_act = 'gelu'
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 512
    type_vocab_size = 2
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    pad_token_id = 0
    position_embedding_type = 'absolute'
    use_cache = True
    classifier_dropout = None



@dataclass
class Data:
    input_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for input sequence"})
    output_max_len: int = field(default=64, metadata={'help': "maximum length to truncate or padded upto for output sequence"})
    data_path_root : Path = field(default=Path('/home/nameduser/Code/babl/inputs'), metadata={'help': "directory path containing fitting data. Should contain files: `10k.jsonl` and `50k.jsonl`"} )
####################################################################################



@dataclass
class Fitting:
    es_patience: int = 2
    model_dir : Path = Path(__file__).parent /  MODELS_CHOICES[os.environ['MODEL_NAME']][0]
    max_epoch: int = 5
    fast_dev_run: bool = False
    mini_dataset: bool = field( default=  True,
        metadata={
            "help": "Whether to use a baby dataset of 128 to test the fitting routine"
        }
    )

    def __post_init__(self):
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

####################################################################################

@dataclass
class Args(Fitting, Data, {'t5':T5,'bloom': Bert, 'bert': Bloom}[os.environ['MODEL_NAME']]):
    def __post_init__(self):
        Fitting.__post_init__(self)

