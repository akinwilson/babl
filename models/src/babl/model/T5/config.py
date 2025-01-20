from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
"""
  1. **TrainingArguments**: These are basicaly the training hyperparameters such as learning rate, batch size, weight decay, gradient accumulation steps etc. See all possible arguments [here](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py). These are used by the Trainer.
  2. **ModelArguments**: These are the arguments for the model that you want to use such as the model_name_or_path, tokenizer_name etc. You'll need these to load the model and tokenizer.
  3. **DataTrainingArguments**: These are as the name suggests arguments needed for the dataset. Such as the directory name where your files are stored etc. You'll need these to load/process the dataset.
"""

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
# @dataclass






@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )



@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    input_dir: str =  field(
        default="inputs",
        metadata={"help": "Directory containing unprocessed input files."},
    )

    # root_dir: str =  field(
    #     metadata={"help": "Root directory "},
    # )


    train_filename: Optional[str] = field(
        default="50k.jsonl",
        metadata={"help": "filename for unprocessed training data"},
    )

    val_filename: Optional[str] = field(
        default="10k.jsonl",
        metadata={"help": "filename for unprocessed validation data"},
    )

    proccessed_train_filename: Optional[str] = field(
        default="train_data.pt",
        metadata={"help": "Path for cached train dataset"},
    )
    proccessed_val_filename: Optional[str] = field(
        default="valid_data.pt",
        metadata={"help": "Path for cached valid dataset"},
    )
    input_max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    output_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    # num_train_epochs: Optional[int] = field(
    #     default=1,
    #     metadata={"help": "num training epochs"},
    # )
