import torch
import json
import logging
import os
from pathlib import Path

from transformers import T5Tokenizer
from .data import prepare_dataset
from argparse import ArgumentParser 

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import T2TDataCollator, ModelArguments, DataArguments


logger = logging.getLogger(__name__)


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    # Check out https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.TrainingArguments
    # to see what other arguments the TrainingArgument accepts 
    args_dict = {
    # "num_cores": 6,
    "model_name_or_path": args.model_name_or_path, #  "t5-small",
    "max_len": args.max_len,
    "target_max_len": args.target_max_len,
    "input_dir":  args.input_dir,
    "output_dir": args.output_dir,
    "overwrite_output_dir": True,
    "per_gpu_train_batch_size": 2,
    "per_gpu_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "tpu_num_cores": 8,
    "do_train": True,
    "num_train_epochs": 32,
    }


    output_dir = Path(args.root_dir) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open( output_dir / "args.json", "w") as f:
        json.dump(args_dict, f)


    input_dir = Path(args.root_dir) /  args.input_dir
    

    model_args, data_args, train_args = parser.parse_json_file(
        json_file= output_dir / "args.json"
    )

    ### THS SHOULD BE PART OF PRIOR STEP OF PIPELINE 
    prepare_dataset(args, data_args)
    ###

       
    if (
        os.path.exists(output_dir)
        and train_args.do_train
        and not train_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        train_args.device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )
    logger.info(f"Training/evaluation parameters:\n{train_args}")

    # Set seed
    set_seed(train_args.seed)

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
            else None
        )
        print("loss: ", loss)
        trainer.save_model()

        tokenizer.save_pretrained(output_dir)

    # Evaluation
    results = {}
    if train_args.do_eval and train_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = output_dir / "eval_results.txt"

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results



"""Start training!"""


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max-len', default=128)
    parser.add_argument('--output-dir', default="outputs")
    parser.add_argument('--input-dir', default="inputs")
    parser.add_argument("--root-dir", default=Path().cwd().parent.parent.parent.parent.parent)
    parser.add_argument('--model-name-or-path', default='t5-small')
    parser.add_argument('--target-max-len', default=32)
    args = parser.parse_args()
    main(args)
