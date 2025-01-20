# Fine-tuning large language models 

## Usage 
To fine-tune a model, please run 
```
python fit.py --model{llama|t5|bloom|bert}
```

Most LLM model architectures that are open sourced are released in a variety of flavours, stratified by their parameter size and therewith capacity. The above fitting rountine has been configured such tha the smaller variants of each architecture are fitted. You can of course change this and choose to train whatever variant you would like to.