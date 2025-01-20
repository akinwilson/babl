# F1: https://en.wikipedia.org/wiki/F-score
## SQuAD evaluation script. Modifed slightly for this notebook
from .config import ModelArguments
from transformers import HfArgumentParser
from .utils import clean

from pathlib import Path
# local data igestion path



parser = HfArgumentParser(ModelArguments)

from collections import Counter
import string
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed
from tqdm.auto import tqdm

set_seed(42)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def test(args):    
    
    model_path = Path(args.output_dir)
    checkpoint =  model_path /  "checkpoint-1" # ""

    # for checkpoint in listdir(checkpoints):
    # model = T5ForConditionalGeneration.from_pretrained(model_path + checkpoint).to("cuda")
    try:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint).to("cuda")
    except:
        print(f"DIDNT LOAD LOCAL TRAINED MODEL: checkpoint:{checkpoint} path didnt work")
        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path).to("cuda")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)#  "t5-small")


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
    print("References:")
    pprint(references)
    print(checkpoint, evaluate(references, predictions))
