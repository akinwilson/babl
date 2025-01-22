import json
import random
from nlp import Dataset
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pytorch_lightning as pl 

from .models import MODELS_CHOICES, MODELS
import os

# from transformers import T5Tokenizer


import torch

import logging

logger = logging.getLogger(__name__)

random.seed(42)




class TextDataset(Dataset):

    def __init__(self, dath_path, tokenizer, plain_text=False, dev_run=True):
        self.dev_run = dev_run
        # tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)  # "t5-small")
        ####################################################################################
        @dataclass
        class DataArgs:
            input_max_len: int = 64
            output_max_len: int = 64

        ####################################################################################
        # super().__init__()
        self.data_args = DataArgs()

        self.data_path = Path(dath_path)
        self.tokenizer = tokenizer
        self.plain_text = plain_text
        self.ds = {}
        if plain_text:
            self.construct_ds(self.extract_valid_pairs(self.read()))
        else:

            self.construct_ds(self.extract_valid_pairs(self.read()))
            self.ds = self.convert_to_features(self.ds)

    def __len__(self):
        return list(self.ds.values())[0].__len__()

    def __getitem__(self, idx):
        if self.plain_text:
            return {
                "input_text": self.ds["input_text"][idx],
                "target_text": self.ds["target_text"][idx],
            }
        else:
            return {
                "input_ids": self.ds["input_ids"][idx],
                "attention_mask": self.ds["attention_mask"][idx],
                "target_ids": self.ds["input_ids"][idx],
                "target_attention_mask": self.ds["attention_mask"][idx],
            }

# 

    def read(self):
        examples = []
        with open(self.data_path, "r") as json_file:
            x = list(json_file)
            # logger.debug(f"[data.py::build_dataset]{x=}")
            for json_str in x:
                examples.append(json.loads(json_str))
        return examples

    def extract_valid_pairs(self, samples):
        valid_questions = []
        for l in samples:
            # clear all docs with more or less than one answer
            # clean all docs with no annotations
            if len(l["annotations"][0]["short_answers"]) == 1:
                if len(l["long_answer_candidates"]) > 2:
                    valid_questions.append(l)
        return valid_questions

    def construct_ds(self, examples):

        self.ds["input_text"] = []
        self.ds["target_text"] = []
  
        for i, q in enumerate(examples):
            # fitting dataset; positive and negative fitting examples
            if random.randint(0, 1) == 1:
                # Construct positive example
                self.ds["input_text"].append(
                    f"question: {q['question_text']}  context: {self.get_long_answer(q)} </s>"[:self.data_args.input_max_len]
                )
                self.ds["target_text"].append(self.get_short_answer(q))
            else:
                # Construct negative example
                self.ds["input_text"].append(
                    f"question: {q['question_text']}  context: {self.get_random_negative(q)} </s>"[:self.data_args.input_max_len]
                )
                self.ds["target_text"].append("None </s>"[:self.data_args.output_max_len])

        if self.dev_run:
            print(f"DEV RUN?: {self.dev_run}. Using 128 datapoints for training")
            self.ds["target_text"] = self.ds["target_text"][:128]
            self.ds["input_text"] = self.ds["input_text"][:128] 

        assert len(self.ds["target_text"]) == len(
            self.ds["input_text"]
        ), "incorrect data distribution"

    def convert_to_features(self, batch):

        input_encodings = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch["input_text"],
            truncation=True,
            pad_to_max_length=True,
            max_length=self.data_args.input_max_len,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch["target_text"],
            truncation=True,
            pad_to_max_length=True,
            max_length=self.data_args.output_max_len,
        )
        # print("input_encodings", input_encodings.keys())
        # print("target_encodings", target_encodings.keys())
        # encodings = {
        #     "input_ids": input_encodings["input_ids"],
        #     "attention_mask": input_encodings["attention_mask"],
        #     "target_ids": target_encodings["input_ids"],
        #     "target_attention_mask": target_encodings["attention_mask"],
        # }

# t5 expects         
# input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, decoder_input_ids=decoder_input_ids
        encodings = {
            "input_ids": torch.tensor(input_encodings["input_ids"]),
            "attention_mask": torch.tensor(input_encodings["attention_mask"]),
            "target_ids": torch.tensor(target_encodings["input_ids"]),
            "target_attention_mask": torch.tensor(target_encodings["attention_mask"]),
        }
        
        return encodings

    def get_exert(self, doc, start_token, end_token):
        return " ".join(doc.split(" ")[start_token:end_token])

    def get_short_answer(self, q):
        answer_indx = q["annotations"][0]["short_answers"][0]
        return self.get_exert(
            q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
        )

    def get_long_answer(self, q):
        answer_indx = q["annotations"][0]["long_answer"]
        return self.get_exert(
            q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
        )

    def get_random_negative(self, q):
        long_answer_indx = q["annotations"][0]["long_answer"]

        for i in range(len(q["long_answer_candidates"])):
            if (
                q["long_answer_candidates"][i]["start_token"]
                == long_answer_indx["start_token"]
            ):
                del q["long_answer_candidates"][i]
                break

        answer_indx = random.choice(q["long_answer_candidates"])
        return self.get_exert(
            q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
        )


class TextDataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer, batch_size=8, dev_run=True):
        super().__init__()
        self.dev_run = dev_run
        self.batch_size = batch_size
        self.train_path = Path(data_path) / "50k.jsonl"
        self.val_path = Path(data_path) / "10k.jsonl"
        # NOTICE, we  re-use the validatio dataset
        self.test_path = Path(data_path) / "10k.jsonl"
        self.tokenizer = tokenizer
        self.pin_memory = False  # True if torch.cuda.is_available() else False

    def train_dataloader(self):
        ds_train = TextDataset(self.train_path, self.tokenizer, dev_run=self.dev_run)
        return DataLoader(
            ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            drop_last=True,
            pin_memory=self.pin_memory,
            collate_fn=T2TDataCollator(),
        )

    def val_dataloader(self):
        ds_val = TextDataset(self.val_path, self.tokenizer, dev_run=self.dev_run)
        return DataLoader(
            ds_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            drop_last=True,
            pin_memory=self.pin_memory,
            collate_fn=T2TDataCollator(),
        )

    def test_dataloader(self):

        ds_test = TextDataset(self.test_path, self.tokenizer, dev_run=self.dev_run)
        return DataLoader(
            ds_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=10,
            drop_last=True,
            pin_memory=self.pin_memory,
            collate_fn=T2TDataCollator(),
        )

# creating a baby dataset for debugging purposes
DEBUG = True


# At the moment, the first variant in the list of models is chosen
# 3 models, there is only one variant, but for t5, there are 5;
# 't5-small', 't5-base', 't5-large','t5-3b','t5-11b'
# A dirt fix would be changing the order of the list, such that whatever
# variant you want train of t5 is first in the list.
# best change this in the ./models.py file


# model_name = [
#     a.default
#     for a in parser._actions
#     if "model-name-or-path" in "".join(a.option_strings)
# ][0]
# tm = MODELS[model_name]

# full_model_name = MODELS_CHOICES[model_name][0]
# args.model_name_or_path = full_model_name


class T2TDataCollator:
    def __call__(self, batch):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([x["input_ids"] for x in batch])
        lm_labels = torch.stack([x["target_ids"] for x in batch])
        
        ####
        # lm_labels[lm_labels[:, :] == 0] = -100
        #### 


        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        decoder_attention_mask = torch.stack(
            [x["target_attention_mask"] for x in batch]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

# def convert_to_features(batch, args, tokenizer):

#     # tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)  # "t5-small")

#     input_encodings = tokenizer.batch_encode_plus(
#         batch["input_text"],
#         truncation=True,
#         pad_to_max_length=True,
#         max_length=args.input_max_len,
#     )
#     target_encodings = tokenizer.batch_encode_plus(
#         batch["target_text"],
#         truncation=True,
#         pad_to_max_length=True,
#         max_length=args.output_max_len,
#     )
#     # print("input_encodings", input_encodings.keys())
#     # print("target_encodings", target_encodings.keys())
#     encodings = {
#         "input_ids": input_encodings["input_ids"],
#         "attention_mask": input_encodings["attention_mask"],
#         "target_ids": target_encodings["input_ids"],
#         "target_attention_mask": target_encodings["attention_mask"],
#     }
#     return encodings


# def prepare_dataset(args, data_args, tokenizer):

#     ## Controls location of input data
#     ##################################################################
#     # train_filename = "50k.jsonl"
#     # val_filename =  "10k.jsonl"

#     input_dir = Path(args.root_dir) / args.input_dir
#     train_path = input_dir / data_args.train_filename
#     test_path = input_dir / data_args.val_filename
#     ##################################################################

#     tds = build_dataset(train_path)
#     vds = build_dataset(test_path)
#     logger.debug("[data.py]::prepare_dataset:finished building ds")

#     txt2feats = partial(convert_to_features, args=args, tokenizer=tokenizer)
#     # map convert_to_features batch wise
#     tds = tds.map(txt2feats, batched=True)
#     # print("tds: ")
#     # pprint(tds)
#     # vds = vds.map(add_eos_to_examples, load_from_cache_file=False)
#     vds = vds.map(txt2feats, batched=True, load_from_cache_file=False)

#     # set the tensor type and the columns which the dataset should return
#     columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
#     tds.set_format(type="torch", columns=columns)
#     vds.set_format(type="torch", columns=columns)

#     # print("tds")
#     # pprint(tds)

#     t_fpath = input_dir / data_args.proccessed_train_filename  # "train_data.pt"
#     v_fpath = input_dir / data_args.proccessed_val_filename

#     torch.save(tds, t_fpath)
#     torch.save(vds, v_fpath)


# def get_exert(doc, start_token, end_token):
#     return " ".join(doc.split(" ")[start_token:end_token])


# def get_short_answer(q):
#     answer_indx = q["annotations"][0]["short_answers"][0]
#     return get_exert(
#         q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
#     )


# def get_long_answer(q):
#     answer_indx = q["annotations"][0]["long_answer"]
#     return get_exert(
#         q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
#     )


# def get_random_negative(q):
#     long_answer_indx = q["annotations"][0]["long_answer"]

#     for i in range(len(q["long_answer_candidates"])):
#         if (
#             q["long_answer_candidates"][i]["start_token"]
#             == long_answer_indx["start_token"]
#         ):
#             del q["long_answer_candidates"][i]
#             break

#     answer_indx = random.choice(q["long_answer_candidates"])
#     return get_exert(
#         q["document_text"], answer_indx["start_token"], answer_indx["end_token"]
#     )


# def build_dataset(data_file):
#     logger.debug(f"[data.py::build_dataset] Hit function. Parameter:{data_file=}")

#     json_lines = []
#     with open(data_file, "r") as json_file:
#         x = list(json_file)
#         logger.debug(f"[data.py::build_dataset]{x=}")
#         for json_str in x:
#             json_lines.append(json.loads(json_str))
#     print("[data.py::build_dataset] finished reading raw data")

#     valid_questions = []
#     for l in json_lines:
#         # clear all docs with more or less than one answer
#         # clean all docs with no annotations
#         if len(l["annotations"][0]["short_answers"]) == 1:
#             if len(l["long_answer_candidates"]) > 2:
#                 valid_questions.append(l)

#     logger.debug(
#         f"[data.py::build_dataset] Num. valid fitting examples: { len(valid_questions)}"
#     )

#     datapoints = {}
#     datapoints["input_text"] = []
#     datapoints["target_text"] = []
#     # datapoints['question']= []
#     # datapoints['question'] = q['question_text']

#     # positive_datapoints = []
#     # negitave_datapoints = []

#     for i, q in enumerate(valid_questions):

#         # fitting dataset; positive and negative fitting examples
#         if random.randint(0, 1) == 1:
#             # Construct positive example
#             datapoints["input_text"].append(
#                 f"question: {q['question_text']}  context: {get_long_answer(q)} </s>"
#             )
#             datapoints["target_text"].append(get_short_answer(q))
#             # if i % 10000 == 0:
#             #     print("-"*100)
#             #     print("Positive fitting example:")
#             #     print(f"[input_text]: question: {q['question_text']}  context: {get_long_answer(q)} </s>")
#             #     print(f"[target_text]: {get_short_answer(q)}")
#             #     print("-"*100)
#         else:
#             # Construct negative example
#             datapoints["input_text"].append(
#                 f"question: {q['question_text']}  context: {get_random_negative(q)} </s>"
#             )
#             datapoints["target_text"].append("None </s>")
#             # if i % 10000 == 0:
#             #     print("-"*100)
#             #     print("negative fitting example:")
#             #     print(f"[input_text]: question: {q['question_text']}  context: {get_random_negative(q)} </s>")
#             #     print(f"[target_text]: None </s>")
#             #     print("-"*100)
#     assert len(datapoints["target_text"]) == len(
#         datapoints["input_text"]
#     ), "incorrect data distribution"

#     # from nlp import Dataset

#     if DEBUG:
#         ds_size = 128
#         logger.debug(
#             f"[data.py::build_dataset]: DEBUG={DEBUG=} --> only using {ds_size} datapoints."
#         )
#         datapoints["input_text"] = datapoints["input_text"][:128]
#         datapoints["target_text"] = datapoints["target_text"][:128]
#         print("")
#     return Dataset.from_dict(datapoints)
