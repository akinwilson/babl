from dataclasses import dataclass


@dataclass
class T5:
    vocab_size : int = 32128
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