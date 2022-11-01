import torch
import json, os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence





def flatten(l):
	return [j for i in l for j in i]

class Consistest(Dataset):
	def __init__(self, args, tokenizer): 
		super(Dataset, self).__init__()
		self.args = args
		self.tokenizer = tokenizer 
		self.samples = []
		self.attributes = []
		self.build()
		
	
	
	def build(self):
		args = self.args
		mem = args.num_history
		datasethash = f'ConsisTest02-h{mem}'  

		if os.path.exists(os.path.join(args.cache_path, datasethash)):
			print(f"Loading dataset from cache ({datasethash})")
			self.samples = json.load(open(os.path.join(args.cache_path, datasethash, 'samples.json')))
			self.attributes = json.load(open(os.path.join(args.cache_path, datasethash, 'attributes.json')))

		else:
			print("Creating dataset ...")
			dialogs = json.load(open(args.dialogs_path))
			persona_qas = json.load(open(args.persona_qas_path))
			history_qas = json.load(open(args.history_qas_path))
			

			sample_idx = 0
			for id, dialog in dialogs.items():
				persona = dialog['persona']
				utterances = dialog['utterances']
				p_qas = flatten([persona_qas.get(p,[]) for p in persona]) 
				h_qas = history_qas.get(id, [])

				for qa in p_qas:
					self.samples.append({'id':sample_idx,'persona':persona, 'history':[qa['q']]})
					self.samples.append({'id':sample_idx + 1,'persona':persona, 'history':utterances[-2*(mem-1):] + [qa['q']]}) 
					self.attributes.append({'id':sample_idx, 'question': qa['q'], 'context':qa['c'], 'response':qa['a'], 'source':qa['source'], 'type':qa['type'], 'dist':0})
					self.attributes.append({'id':sample_idx + 1, 'question': qa['q'], 'context':qa['c'], 'response':qa['a'], 'source':qa['source'], 'type':qa['type'], 'dist':mem-1})
					sample_idx +=2
				
				for qa in h_qas:
					for dist in [0,2]:    # distance between question and fact
						assert dist <= mem
						insert_loc = 2*(qa['turn'] + dist + 1)
						if insert_loc <= len(utterances)+1:
							history = utterances[:insert_loc][-2*mem:] + [qa['q']]
							self.samples.append({'id':sample_idx, 'persona':persona, 'history':history})
							self.attributes.append({'id':sample_idx, 'question': qa['q'],'context': qa['c'], 'alt_context':qa['alt_c'], 'response':qa['a'], 'source':qa['source'], 'type':qa['type'], 'dist':dist})
							sample_idx +=1

			assert sample_idx == len(self.samples)

			os.makedirs(os.path.join(args.cache_path, datasethash))
			json.dump(self.samples, open(os.path.join(args.cache_path, datasethash, 'samples.json'), 'w'))
			json.dump(self.attributes, open(os.path.join(args.cache_path, datasethash, 'attributes.json'), 'w'))




	def add_new_tokens(self):
		NEW_TOKENS = {'additional_special_tokens': self.args.model_sep_tokens}
		self.tokenizer.add_special_tokens(NEW_TOKENS) 

	
	def __len__(self) -> int:
		return len(self.samples)



	def add_tokens_to_sample(self, sample):
		sp_id = self.args.model_sep_tokens
		agent_id = sp_id[1]
		history, persona = sample['history'], sample['persona']
		persona = f"{agent_id} {' '.join(persona)}"  
		history.reverse()
		history = [f"{sp_id[i%2]} {h}" for i,h in enumerate(history)]
		history.reverse()
		return history, persona

	
	
	def __getitem__(self, index: int):
		args = self.args
		sample = self.samples[index]
		tok = self.tokenizer
		history, persona = self.add_tokens_to_sample(sample)
		history_ids = [tok.encode(h, add_special_tokens=False, max_length= args.utterance_max_length, truncation=True) for h in history]
		persona_ids = tok.encode(persona)[:-1]
		input_ids = persona_ids + flatten(history_ids) + [tok.eos_token_id]
		return {'input_ids':torch.tensor(input_ids), 'idx':sample['id']}



	def collate(self, batch):
		input_ids = [b['input_ids'] for  b in batch]
		input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
		output = {"input_ids":input_ids, 'idx': [b['idx'] for  b in batch]}
		return output
