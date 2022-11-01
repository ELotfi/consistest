import json
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from datasets import Dataset as HFDset

from consis_class import Consistest
from util.hybrid import check_rules, clean_response




# returns the NLI score using the DNLI model
def evaluate_nli(args, df):
	model = RobertaForSequenceClassification.from_pretrained(args.nli_model)
	tokenizer = RobertaTokenizer.from_pretrained(args.nli_model)

	dataset_nli = HFDset.from_pandas(df[['NLI_Ref','Response','Clean_Resp', 'Reference', 'Type', 'Question']])
	dataset_nli = dataset_nli.map(lambda x: tokenizer(x['NLI_Ref'],x['Clean_Resp']), batched=True, remove_columns=dataset_nli.column_names)
	trainer = Trainer(model=model, tokenizer=tokenizer)
	predictions = trainer.predict(dataset_nli).predictions
	df['nli_labels'] = np.argmax(predictions, axis=1)
	df['nli_score'] = (df['nli_labels']==2).astype(int)
	return df


# return final score based on the token-level and nli scores
def finalize_score(rule, nli, q_type):
	if rule == -1:
		return int(nli==2)
	elif rule == 3:
		return int(nli!=0)
	elif q_type == 'wh' and nli == 0:
		return nli
	else: return rule



# runs the hybrid evaluation pipeline
def evaluate_hyb(args, df):
	if 'nli_labels' not in df.columns:
		df = evaluate_nli(args, df)
	df['rule_scores'] = df.apply(lambda x: check_rules(x), axis=1)
	df['hyb_score'] = df.apply(lambda x: finalize_score(x['rule_scores'], x['nli_labels'], x['Type']), axis=1)
	return df



	
def report_scores(df):
	res = {}
	col = 'hyb_score'
	res['All'] = df[col].mean()
	res['YN'] = df[df['Type']=='yn'][col].mean()
	res['WH'] = df[df['Type']=='wh'][col].mean()
	res['Pers'] = df[df['Source']=='pers'][col].mean()
	res['His'] = df[df['Source']=='hist'][col].mean()
	res['His_ext'] = df[df['Source']=='hist_ext'][col].mean()

	subres = {'persona':{}, 'history':{}} 
	subp = df[df['Source'] == 'pers']
	subh = df[df['Source'] != 'pers']
	for t in ['yn', 'wh']:
		subres['persona'][t] = subp[subp['Type']==t]['hyb_score'].mean()
		subres['history'][t] = subh[subh['Type']==t]['hyb_score'].mean()
	return res, subres




def evaluate(args, df):
	df = df.apply(clean_response, axis=1)
	df = evaluate_hyb(args, df)
	mean_scores, subset_scores = report_scores(df)
	return df, mean_scores, subset_scores



# generate responses to questions 
def infer(args, dataloader, model, tokenizer):
	outputs = []
	device = args.device
	model.to(device)
	model.eval()
	start_token_id =tokenizer.convert_tokens_to_ids('<pad>') if 't5' in model.config._name_or_path else model.config.decoder_start_token_id

	for batch in tqdm(dataloader):
		with torch.no_grad():
			input_ids = batch['input_ids'].to(device)
			attention_mask = (input_ids != tokenizer.pad_token_id).long()
			response = model.generate(input_ids,
									  attention_mask=attention_mask, 
									  max_length=25, 
									  num_beams=args.n_beams, 
									  temperature = args.temp,
									  top_p = args.top_p,
									  decoder_start_token_id=start_token_id)
			response = tokenizer.batch_decode(response, skip_special_tokens=True)
			outputs += [{'idx':i, 'response':r} for i,r in zip(batch['idx'], response)]
	
	return outputs




def main(args):
	if args.eval_only:
		print("Evaluating ...")
		model_outputs = pd.read_csv(args.responses_to_eval)
		df, mean_scores, subset_scores = evaluate(args, model_outputs)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
		dataset = Consistest(args, tokenizer)
		reference = dataset.attributes
		loader = DataLoader(dataset, collate_fn=dataset.collate, batch_size=args.batch_size)
		print("Generating responses ...")		
		model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
		model.eval()
		responses = infer(args, loader, model, tokenizer)
		df = pd.DataFrame({
							'Question':[r['question'] for r in reference], 
							'Response':[r['response'] for r in responses],
							'Reference':[r['response'] for r in reference],
							'NLI_Ref':[(r.get('alt_context', "") if r.get('alt_context', "") != "" else r['context']) for r in reference],
							'Source':[r['source'] for r in reference], 
							'Type':[r['type'] for r in reference], 
							'Distance':[r.get('dist', -1) for r in reference]})
		df.to_csv(f"{args.results_path}/model_output.csv", index=False)
		print("Evaluating responses ...")
		df, mean_scores, subset_scores = evaluate(args, df)

	results = {'scores':mean_scores, 'subsets':subset_scores}
	df.to_csv(f"{args.results_path}/model_eval.csv", index=False)
	json.dump(results, open(f"{args.results_path}/eval_summary.json", 'w'))




if __name__ == '__main__':
	parser = ArgumentParser()
	# dataset parameters
	parser.add_argument("--cache_path", default="cache/")
	parser.add_argument("--dialogs_path", default="data/valid.json")
	parser.add_argument("--persona_qas_path", default="data/persona_qas.json")
	parser.add_argument("--history_qas_path", default="data/history_qas.json")
	parser.add_argument("--num_history", default=3)

	# eval parameters
	parser.add_argument("--nli_model", default="Ehsanl/Roberta-DNLI")
	parser.add_argument("--model_checkpoint")  
	parser.add_argument("--model_sep_tokens", default=["<user>", "<agent>" ])  # special tokens used to separate user and agent utterances in training   
	parser.add_argument("--eval_only", default=False)
	parser.add_argument("--responses_to_eval", default='results/model_output.csv')
	parser.add_argument("--results_path", default="results")
	parser.add_argument("--utterance_max_length", default=32)
	parser.add_argument("--batch_size", default=16)
	parser.add_argument("--device", default=torch.device("cuda:0"))

	# generation parameters
	parser.add_argument("--n_beams", default=1)
	parser.add_argument("--temp", default=1.)
	parser.add_argument("--top_p", default=1.)
	parser.add_argument("--decoding", default="greedy", choices=["greedy", "beam"])


	args = parser.parse_args()
	main(args)
