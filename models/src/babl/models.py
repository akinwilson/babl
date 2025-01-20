from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer


MODELS_CHOICES = {
    "t5": ['t5-small', 't5-base', 't5-large','t5-3b','t5-11b'],
    "llama": ['meta-llama/Llama-3.3-70B-Instruct'],
    "bert": ['google-bert/bert-base-uncased'],
    "bloom": ["bigscience/bloom"]}
# just choosing smallest t5 model for now 
MODELS = { 
    "t5": {"tok": T5Tokenizer, "model": T5ForConditionalGeneration},
    "llama":{"tok": AutoTokenizer, "model":AutoModelForCausalLM} ,
    "bert": {"tok":AutoTokenizer, "model":AutoModelForMaskedLM},
    "bloom": {"tok":AutoTokenizer, "model":AutoModelForCausalLM}}
