# Babl webserver application 

## Overview 
The code accompanying this folder is a django webserver application which allows endusers to interact with a selection of large language models; [T5](https://en.wikipedia.org/wiki/T5_(language_model)), [LLaMA](https://en.wikipedia.org/wiki/Llama_language_model) and 
[Bloom](https://en.wikipedia.org/wiki/BLOOM_(language_model))


## Running tests 

## Starting your own project 


Create a virtual environment and install the python requirements
```
pip install -r requirements.txt
```
Create a directory with you application name; e.g. `babl` and start django project with 
```
django-admin startproject core babl
``` 
`core` refers to the core settings of the django project. 

Then run your own project in development mode with:
```
python babl/manage.py runserver 127.0.0.1:7070
```





