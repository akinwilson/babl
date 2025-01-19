import json
import random
from nlp import Dataset

random.seed(42)


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
