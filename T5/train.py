from re import T
import tqdm
import torch
import nlp
from transformers import T5Tokenizer
try:
    from .read import build_dataset
except:
    from read import build_dataset

import json
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch["input_text"], pad_to_max_length=True, max_length=512
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["target_text"], pad_to_max_length=True, max_length=16
    )

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }

    return encodings

from pathlib import Path

ext_train = "data/50k.jsonl"
ext_val =  "data/10k.jsonl"

try:
    cloud_train_path = f"/gcs/ml-operations/{ext_train}"
    cloud_val_path = f"/gcs/ml-operations/{ext_val}"
    train_dataset = build_dataset(cloud_train_path)
    valid_dataset = build_dataset(cloud_val_path)
except:
    train_path = str(Path(__file__).parent.resolve() / ext_train )
    test_path = str(Path(__file__).parent.resolve() / ext_val )
    train_dataset = build_dataset(train_path)
    valid_dataset = build_dataset(test_path)


# map convert_to_features batch wise
train_dataset = train_dataset.map(convert_to_features, batched=True)

# valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
valid_dataset = valid_dataset.map(
    convert_to_features, batched=True, load_from_cache_file=False
)

# set the tensor type and the columns which the dataset should return
columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
train_dataset.set_format(type="torch", columns=columns)
valid_dataset.set_format(type="torch", columns=columns)


t_fname = "train_data.pt"
v_fname = "valid_data.pt"

try:
# cache the dataset, so we can load it directly for training
    cloud_train_path = f"/gcs/ml-operations/{t_fname}"
    cloud_val_path = f"/gcs/ml-operations/{v_fname}"
    torch.save(train_dataset, cloud_train_path)
    torch.save(valid_dataset, cloud_val_path)
except:
    torch.save(train_dataset, t_fname)
    torch.save(valid_dataset, v_fname)


from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

from data_classes import T2TDataCollator, ModelArguments, DataTrainingArguments

args_dict = {
    "num_cores": 6,
    "training_script": "train.py",
    "model_name_or_path": "t5-small",
    "max_len": 128,
    "target_max_len": 16,
    "output_dir": "/gcs/ml-operations/models/gpu",
    "overwrite_output_dir": True,
    "per_gpu_train_batch_size": 2,
    "per_gpu_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "tpu_num_cores": 8,
    "do_train": True,
    "num_train_epochs": 32,
}

with open("args.json", "w") as f:
    json.dump(args_dict, f)



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath("args.json")
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
    )

    # Training
    if training_args.do_train:
        loss = trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        print("loss: ", loss)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results



"""Start training!"""
if __name__ == "__main__":
    main()
