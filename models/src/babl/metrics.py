# F1: https://en.wikipedia.org/wiki/F-score
## SQuAD evaluation script. Modifed slightly for this notebook
from .model.T5.config import ModelArguments
from transformers import HfArgumentParser
from .utils import clean
from pathlib import Path

# local data igestion path
from collections import Counter
from .utils import normalize_answer
import torch
from transformers import set_seed
from tqdm.auto import tqdm


import logging

logger = logging.getLogger(__name__)


set_seed(42)

parser = HfArgumentParser(ModelArguments)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truth, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += exact_match_score(prediction, ground_truth)
        f1 += f1_score(prediction, ground_truth)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def test(args, model, tokenizer):

    model_path = Path(args.output_dir)

    # again assuming its always checkpoint-1 we wish to load
    checkpoint_path = model_path / "checkpoint-1"  # ""
    logger.debug(f"[metrics.py::test]: {checkpoint_path=}")
    # for checkpoint in listdir(checkpoints):
    # model = T5ForConditionalGeneration.from_pretrained(model_path + checkpoint).to("cuda")
    try:
        model = model.from_pretrained(checkpoint_path).to("cuda")
    except:
        logger.debug(
            f"DIDNT LOAD LOCAL TRAINED MODEL: checkpoint:{checkpoint_path} path didnt work"
        )
        model = model.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path
        ).to("cuda")

    # tokenizer = tokenizer.from_pretrained(args.model_name_or_path)#  "t5-small")

    val_path = Path(args.input_dir) / "valid_data.pt"
    val_ds = torch.load(val_path)
    dl = torch.utils.data.DataLoader(val_ds, batch_size=28)

    answers = []
    for batch in dl:
        outs = model.generate(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
            max_length=16,
            early_stopping=True,
        )
        outs = [tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)

    predictions = []
    references = []
    for ref, pred in zip(val_ds, answers):
        predictions.append(clean(pred))
        references.append(clean(tokenizer.decode(ref["target_ids"])))
    from pprint import pprint

    logger.debug(f"[{__file__}::test ] References:")
    logger.debug(references)
    print(checkpoint_path, evaluate(references, predictions))
