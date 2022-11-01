# consistest
A conversational benchmark for factual consistency

This is the repository for the paper: **What Was Your Name Again? Interrogating Generative Conversational Models For Factual Consistency Evaluation**

## Instructions
With the code you can evaluate the factual consistency for a generative model using the dataset and hybrid pipeline described in paper. 
### Evaluate a model
1. Finetune the model on PersonaChat train set (not necessary but for a fair comparisson). 
2. Run: `python evaluate.py --model_checkpoint={path to your model} --model_sep_tokens={your model's special tokens}`. The second parameter is the list of potential special tokens used in finetuning to separate agent and user utterances, and defaults to ["\<user\>", "\<agent\>"].
3. The code first curates the Consistest dataset and then runs inference and evaluation. The results are saved in the `results` folder.
Note that the code assumes your model and tokenizer can be read using the `.from_pretrained()` method in `transformers` library.
### Evaluate responses
You can also de the evaluation on already generated responses (as a `.csv` file) by running: `python evaluate.py --eval_only=True --responses_to_eval={path to response file}`.
Note that the `.csv` file should follow the expected format; i.e. `Question`, `Response`, `Reference`, `NLI_Ref`, `Source`, `Type`, `Distance` as columns.

---
In both cases since the code uses the `trainer` method from `transformers` for NLI inference, all available GPUs will be used. 
This can be changed with `CUDA_VISIBLE_DEVICES={list of GPU ids}` in front of the `python evaluate.py` line.
