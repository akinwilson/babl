# Deploying in product 

HuggingFace is a library offering pre-trained transformer networks that can be fine tuned for down stream task such as, for example: 
- Sentiment Analysis 
- Summarisation 
- Information Retrival
- Entity Extraction 
- Entity Linking 


This repository will consider the application of the transformer networks to the task of **question and answering**; a current research domain of deep learning that is still not considered an *AI complete task* (for example, in the domain of computer vision, image classification, semantic segmentation and so on are considered *AI complete* since they have surpassed human accurary) 

The example giving will fine tune the T5 transformer network on a question and answering task with the addition of providing additional context to the encoder to prove or disprove **if** the additional contexts improves the overall performance of the network. 

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

