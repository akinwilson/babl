# Fine-tuning large language models 


## Usage 

To fine-tune a model outside of a container, the following command pattern can be used.
```
export MODEL_NAME={llama|t5|bloom|bert} && python fit.py 
```
where `{llama|t5|bloom|bert}` needs to be changed to one model, so for running bert, run
```
export MODEL_NAME=bert && python fit.py
```
*why have both command line and environment variables as inputs?*
<br>
<br>
Later on, you may wish to perform hyperparameter optmisation, and wish to pass the hyperparameters via the command line. When not performing an experiment and you want to run the fitting script straight at the box, then provide the model you wish to fit-tune via exporting its name as shown above.  

Most LLM model architectures that are open sourced are released in a variety of flavours, stratified by their parameter count and therewith capacity. The above fitting rountine has been configured such tha the smaller variants of each architecture are fitted. You can of course change this and choose to train whatever variant you would like to.