# Fitting large language models

## Overview 

## Models 

The bellow table provides a summary of the models available in `babl` library. 

| Model     |               Task |                       Size (param count) |                              Training config  | Architecture | EM | 
| ----------- |        ----------- | -----------                |                          ----------- |----------- | - |
| Bloom               | sequence to sequence                  |    ~176B         |  Causal/Autoregressive              | Transformer | 93% | 3% | 
|  T5  | sequence to sequence     |    ~220M/11B         | Masked     | Transformer| 90% | 2%|  91% |
|   LLama  | sequence to sequence    |    ~7B/70B         | Causal/Autoregressive     | Transformer | 80% 
| BERT | sequence to sequence | ~29M/334M | Masked | Transformer | 77%
