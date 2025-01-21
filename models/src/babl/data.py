import json
import random
from nlp import Dataset
from pathlib import Path 
from functools import partial
from transformers import T5Tokenizer
import torch 

import logging 
logger = logging.getLogger(__name__)

random.seed(42)

# creating a baby dataset for debugging purposes 
DEBUG = True 

class T2TDataCollator:
    def __call__(self, batch):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        
        input_ids = torch.stack([x["input_ids"] for x in batch])
        lm_labels = torch.stack([x["target_ids"] for x in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        decoder_attention_mask = torch.stack([x["target_attention_mask"] for x in batch])
        ####### NOTICE WE HAVE LABELS instead of TARGET_IDS the model expects these as inputs 
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }





def convert_to_features(batch, args):
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path) # "t5-small")
    input_encodings = tokenizer.batch_encode_plus(batch["input_text"],truncation=True, pad_to_max_length=True, max_length=args.input_max_len)
    target_encodings = tokenizer.batch_encode_plus(batch["target_text"],truncation=True, pad_to_max_length=True, max_length=args.output_max_len)
    # print("input_encodings", input_encodings.keys())
    # print("target_encodings", target_encodings.keys())
    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "target_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }
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

    tds = build_dataset(train_path)
    vds = build_dataset(test_path)
    logger.debug("[data.py]::prepare_dataset:finished building ds")

    txt2feats = partial(convert_to_features, args=args)
    # map convert_to_features batch wise
    tds = tds.map(txt2feats, batched=True)
    # print("tds: ")
    # pprint(tds)
    # vds = vds.map(add_eos_to_examples, load_from_cache_file=False)
    vds = vds.map(
        txt2feats, batched=True, load_from_cache_file=False
    )

    # set the tensor type and the columns which the dataset should return
    columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
    tds.set_format(type="torch", columns=columns)
    vds.set_format(type="torch", columns=columns)
    
    # print("tds")
    # pprint(tds)

    t_fpath = input_dir / data_args.proccessed_train_filename # "train_data.pt"
    v_fpath = input_dir / data_args.proccessed_val_filename

    torch.save(tds, t_fpath)
    torch.save(vds, v_fpath)




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
        if (q["long_answer_candidates"][i]["start_token"] == long_answer_indx["start_token"]):
            del q["long_answer_candidates"][i]
            break

    answer_indx = random.choice(q["long_answer_candidates"])
    return get_exert(
        q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
    )


def build_dataset(data_file):
    logger.debug(f"[data.py::build_dataset] Hit function. Parameter:{data_file=}")

    json_lines = []
    with open(data_file, "r") as json_file:
        x = list(json_file)
        logger.debug(f"[data.py::build_dataset]{x=}")
        for json_str in x:
            json_lines.append(json.loads(json_str))
    print("[data.py::build_dataset] finished reading raw data")

    
    valid_questions = []
    for l in json_lines:
        # clear all docs with more or less than one answer
        # clean all docs with no annotations
        if len(l["annotations"][0]["short_answers"]) == 1:
            if len(l["long_answer_candidates"]) > 2:
                valid_questions.append(l)

    logger.debug("[data.py::build_dataset] Num. examples:", len(valid_questions))

    datapoints = {}
    datapoints["input_text"] = []
    datapoints["target_text"] = []
    # datapoints['question']= []
    # datapoints['question'] = q['question_text']

    # positive_datapoints = []
    # negitave_datapoints = []
    
    for (i,q) in enumerate(valid_questions):

        # fitting dataset; positive and negative fitting examples 
        if random.randint(0, 1) == 1:
            # Construct positive example
            datapoints["input_text"].append(f"question: {q['question_text']}  context: {get_long_answer(q)} </s>")
            datapoints["target_text"].append(get_short_answer(q))
            # if i % 10000 == 0:
            #     print("-"*100)
            #     print("Positive fitting example:")
            #     print(f"[input_text]: question: {q['question_text']}  context: {get_long_answer(q)} </s>")
            #     print(f"[target_text]: {get_short_answer(q)}")
            #     print("-"*100)
        else:
            # Construct negative example
            datapoints["input_text"].append(f"question: {q['question_text']}  context: {get_random_negative(q)} </s>")
            datapoints["target_text"].append("None </s>")
            # if i % 10000 == 0:
            #     print("-"*100)
            #     print("negative fitting example:")
            #     print(f"[input_text]: question: {q['question_text']}  context: {get_random_negative(q)} </s>")
            #     print(f"[target_text]: None </s>")
            #     print("-"*100)
    assert len(datapoints["target_text"]) == len(datapoints["input_text"]), "incorrect data distribution"

    # from nlp import Dataset


    if DEBUG:
        ds_size = 128
        logger.debug(f"[data.py::build_dataset]: DEBUG={DEBUG=} --> only using {ds_size} datapoints.")
        datapoints["input_text"] = datapoints["input_text"][:128] 
        datapoints["target_text"] = datapoints["target_text"][:128]
        print("")
    return Dataset.from_dict(datapoints)
