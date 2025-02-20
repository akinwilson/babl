# babl

![alt text](img/parrot.jpg "conversational")


## Overview 
Causally and masked pretrained deep learning networks have been applied to a variety of domains, particularly [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). Babl is a library which allows users to interact with a collection of large language models locally through the web browser. Babl also demonstrates how to [fine-tune](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) these models on down-stream applications. This fine-tuning is performed on a question and answering dataset create from Wikipedia by Standford academics; [SQuAD](https://arxiv.org/abs/1606.05250). Babl provides access to four models:
1) [T5](https://en.wikipedia.org/wiki/T5_(language_model))
2) [LLaMA](https://en.wikipedia.org/wiki/Llama_language_model)
3) [Bloom](https://en.wikipedia.org/wiki/BLOOM_(language_model))
4) [BERT](https://en.wikipedia.org/wiki/BERT_(language_model))


## Usage
To prop up all services; the fine-tuning of the models, model serving and webserver application, run

```
docker-compose -f docker-compose.yaml
```

Once the fine-tuning job has been completed (which should be apparent from the logs), the serving container will have a [interactive API documentation page](https://fastapi.tiangolo.com/features/#based-on-open-standards) accessible for end users to test out the inference API directly. This will be accessible from:

```
http://localhost:6000/docs
```

there will also be a webserver serving an application allowing end users to test out the inference API via recording their own clips and posting them to the API through a simple user interface. This will be accessible from:
```
http://localhost:7000
```

## Installation 

Create a virtual environment with your preferred environment manage and install the `python` package requirements. 
```
pip install -r requirements.txt
```
To intstall the python library containing all the large language models, `babl`, run 
```
pip install -e ./models
```
If you would like to run the `fit/fit.py` script to train a model and need the data, run the following `bash` script to download it: 
```
./pull_data.sh
```

## Running tests



## Citation 



## What is SQuAD?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

This model fine tunes on this data set, i.e. `inputs/50k.jsonl` corresponds to the SQuAD training dataset. 

Often in various domains there is a standard dataset all ml engineers compare their state of the art models against to make the model comparisions fair. For question and answer that dataset is the SQuAD dataset. 


## Data 
A dataset has been created from Wikipedia. 
- it contains questions (**x**) and their ground truth answers (**y**)
- Alongside side these pairs (**x**, **y**), the dataset contains context **c** deemed relevant to answering the underlying question 
- i.e. Sample example ((**x**, **c**), **y**). 

## Experiment

ALongside training and delopying the models, this library implements an experiment which, once conducted, shouls provide motivation as to how and why the [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) architecture was developed.  The goal of the experiment is to prove or disprove if providing the encoder of the network with the additional context **c** alongside the question **x** improves the overall performance of the generator (the component of the network that produces **y**)

- I.e. will the performance metrics improve, w.r.t the second model trained in dataset 2), if two models are fined tuned on two different datasets:
1) Dataset: (**x**,**y**)
2) Dataset: ([**x**,**c**],**y**)
- Where we concatenate the question **x** with the context **c** in the latter dataset. 

## Todo 20 Jan 2025

**LAST STATE** T5 training. with `export MODEL_NAME=t5 && pthon fit.py`. Need to containerised. Want ot be able to provide MODEL_NAME as env var, and the provide command `python fit.py -h` into the fitting container to show model-specific as well as fitting and data-related  parameter options. need to write bash script for entrypoing like 

a `start.sh` like  
```bash 
#!/bin/bash 
exec "@"
```

as ENTRYPOINT in dockefile such that  the fitting can be run it like (executed from the root of this directory) 

```
cd fit && docker build . -f Dockerfile.fit fit:latest && cd ..
docker run --name fit -v $(pwd)/inputs:/usr/app/src/inputs -e MODEL_NAME=t5  fit:latest  python fit.py -h  
```


which should then print parameter 

 Was just agregrating all the parameters in `config.py` and successfully used there was an [issue](https://github.com/mivade/argparse_dataclass/issues/65) surrounding printing help from the cli. Also, you were figuring out [how to merge](https://stackoverflow.com/questions/79380660/how-to-pythonically-merging-python-data-classes-with-unique-attribute-names) dataclasses in prepartion for using with [dataclass_argparse](https://pypi.org/project/argparse-dataclass/). Dont forget, you wanted to also figured out how to split the parsed data back into instances of the dataclasses whcich they belong to. 

- [ ] factor metric caclculations into own mondule. 
- [ ] Test fiting for every model. You can now fit each model with a single command ; `python fit.py --model-name-or-path {t5,llama,bloom,bert}`
- [ ] Containerised the serving container and test along with fine-tuning job and application webserver the end to end application 
- [ ] Developed frontend in chatapp application. Can used [this project](https://github.com/meghsohor/chat-app-vanilla-js/tree/master) as a guide
- [ ] Store chat conversations on the backend with djangos database integration. 
- [x] locally train and query T5 model from the command line 
- [x] retrieve bloom, LLaMA and BERT models and apply the same
- [x] develop the serving framework, using fastAPI.  
- [x] Django full-stack (https://dev.to/documatic/build-a-chatbot-using-python-django-46hb) to allow for interaction with models, storing and retriewing of conversations
- [ ] Write tests 
- [x] installable python package
- [ ] Automated testing with GH actions 
