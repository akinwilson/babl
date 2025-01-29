from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(
    model_args.tokenizer_name
    if model_args.tokenizer_name
    else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)

model = T5ForConditionalGeneration.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)