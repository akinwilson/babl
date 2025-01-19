import json
import random
from nlp import Dataset
from pathlib import Path 
from functools import partial
from transformers import T5Tokenizer
import torch 


random.seed(42)

def convert_to_features(example_batch, args):
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path) # "t5-small")
    input_encodings = tokenizer.batch_encode_plus( 
        example_batch["input_text"],truncation=True, pad_to_max_length=True, max_length=args.input_max_len
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["target_text"],truncation=True, pad_to_max_length=True, max_length=args.output_max_len
    )
    # print("input_encodings", input_encodings.keys())
    # print("target_encodings", target_encodings.keys())
    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }
    # print("encodings['target_ids']: ", encodings['target_ids'][:1])
    return encodings




def prepare_dataset(args, data_args):


    ## Controls location of input data
    ##################################################################
    # train_filename = "50k.jsonl"
    # val_filename =  "10k.jsonl"

    input_dir = Path(args.root_dir) /  args.input_dir
    train_path = input_dir / data_args.train_filename 
    test_path = input_dir / data_args.val_filename 
    ##################################################################

    train_dataset = build_dataset(train_path)
    valid_dataset = build_dataset(test_path)


    txt2feats = partial(convert_to_features, args=args)
    # map convert_to_features batch wise
    train_dataset = train_dataset.map(txt2feats, batched=True)

    # valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
    valid_dataset = valid_dataset.map(
        txt2feats, batched=True, load_from_cache_file=False
    )

    # set the tensor type and the columns which the dataset should return
    columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)


    t_fpath = input_dir / data_args.proccessed_train_filename # "train_data.pt"
    v_fpath = input_dir / data_args.proccessed_val_filename

    torch.save(train_dataset, t_fpath)
    torch.save(valid_dataset, v_fpath)




def get_exert(doc, start_token, end_token):
    return " ".join(doc.split(" ")[start_token:end_token])


def get_short_answer(q):
    answer_indx = q["annotations"][0]["short_answers"][0]
    return get_exert(
        q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
    )


def get_long_answer(q):
    answer_indx = q["annotations"][0]["long_answer"]
    return get_exert(
        q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
    )


def get_random_negative(q):
    long_answer_indx = q["annotations"][0]["long_answer"]

    for i in range(len(q["long_answer_candidates"])):
        if (
            q["long_answer_candidates"][i]["start_token"]
            == long_answer_indx["start_token"]
        ):
            del q["long_answer_candidates"][i]
            break

    answer_indx = random.choice(q["long_answer_candidates"])
    return get_exert(
        q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
    )


def build_dataset(data_file):

    with open(data_file, "r") as json_file:
        json_list = list(json_file)

    json_lines = []
    for json_str in json_list:
        json_lines.append(json.loads(json_str))

    valid_questions = []
    for l in json_lines:
        # clear all docs with more or less than one answer
        # clean all docs with no annotations
        if len(l["annotations"][0]["short_answers"]) == 1:
            if len(l["long_answer_candidates"]) > 2:
                valid_questions.append(l)

    print("num valid:", len(valid_questions))

    datapoints = {}
    datapoints["input_text"] = []
    datapoints["target_text"] = []
    # datapoints['question']= []
    # datapoints['question'] = q['question_text']

    positive_datapoints = []
    negitave_datapoints = []
    for q in valid_questions:

        # train
        if random.randint(0, 1) == 1:
            # Construct positive example
            datapoints["input_text"].append(
                f"question: {q['question_text']}  context: {get_long_answer(q)} </s>"
            )
            datapoints["target_text"].append(get_short_answer(q))
        else:
            # Construct negative example
            datapoints["input_text"].append(
                f"question: {q['question_text']}  context: {get_random_negative(q)} </s>"
            )
            datapoints["target_text"].append("None </s>")

    assert len(datapoints["target_text"]) == len(
        datapoints["input_text"]
    ), "incorrect data distribution"

    # from nlp import Dataset
    return Dataset.from_dict(datapoints)
