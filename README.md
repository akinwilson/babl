# Babl

![alt text](img/parrot.jpg "conversational")
# Todo 21 Dec 2024
- [] locally train and query T5 model from the command line 
- [] retrieve bloom and LLaMA models and apply the same 
- [] Django full-stack (https://dev.to/documatic/build-a-chatbot-using-python-django-46hb) to allow for interaction with models, storing and retriewing of conversations
- [] Write tests 
- [] installable python package
- [] Automated testing with GH actions 

## Overview 
Causally and masked pretrained deep learning networks have been applied to a variety of domains, particularly [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). Babl is a library which allows users to interact with a collection of large language models locally through the web browser. Babl also demonstrates how to [fine-tune](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) these models on down-stream applications. This fine-tuning is performed on a question and answering dataset create from Wikipedia by Standford academics; [SQuAD](https://arxiv.org/abs/1606.05250). Babl provides access to three models:
1) [T5](https://en.wikipedia.org/wiki/T5_(language_model))
2) [LLaMA](https://en.wikipedia.org/wiki/Llama_language_model)
3) [Bloom](https://en.wikipedia.org/wiki/BLOOM_(language_model))



## Installation 

Create a virtual environment and install the requirements
```
pip install -r requirements.txt
```
Run the script to download the training data: 
```
./pull_data.sh
```

## Usage
To prop up all services; the fine-tuning of the models, model serving and webserver application, run

```
docker-compose -f docker-compose.yaml
```

To fit outside of a container, run 

```
pip install -r ./models 
```
and 
```
python fit/train.py
```


## Running tests


## Citation 



# What is SQuAD?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

This model fine tunes on this data set, i.e. `inputs/50k.jsonl` corresponds to the SQuAD training dataset. 

Often in various domains there is a standard dataset all ml engineers compare their state of the art models against to make the model comparisions fair. For question and answer that dataset is the SQuAD dataset. 


# Data 
A dataset has been created from Wikipedia. 
- it contains questions (**x**) and their ground truth answers (**y**)
- Alongside side these pairs (**x**, **y**), the dataset contains context **c** deemed relevant to answering the underlying question 
- i.e. Sample example ((**x**, **c**), **y**). 

# Experiment goal
- The goal of the experiment is to prove or disprove if providing the encoder of the network with the additional context **c** alongside the question **x** improves the overall performance of the generator (the component of the network that produces **y**)

- I.e. will the performance metrics improve, w.r.t the second model trained in dataset 2), if two models are fined tuned on two different datasets:
1) Dataset: (**x**,**y**)
2) Dataset: ([**x**,**c**],**y**)
- Where we concatenate the question **x** with the context **c** in the latter dataset. 
