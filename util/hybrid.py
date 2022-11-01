
from nltk.tokenize import word_tokenize, RegexpTokenizer
import json, re
from nltk.metrics import recall
from util.w2n import word_to_num




short_answers = {'do you': {'pos': ['i do'], 'neg': ["i don't", 'i do not']},
					'are you': {'pos': ['i am', "i'm"], 'neg': ["i'm not", 'i am not']},
					'can you': {'pos': ['i can'], 'neg': ["i can't", 'i can not']},
					'did you': {'pos': ['i did'], 'neg': ["i didn't", 'i did not']},
					'will you': {'pos': ['i will'], 'neg': ['i will not', "i won't"]},
					'have you': {'pos': ['i have', "i've"], 'neg': ['i have not', "i haven't"]}
				}
neg_contracts = {"have":"'ve ", "are":"'re ", "will":"'ll ", "am":"'m ", "is":"'s ", "can":" ca", "will":" wo"}	
pos_contracts = {"have":"'ve", "are":"'re", "will":"'ll", "am":"'m", "is":"'s"}	

yns = {1:['yes', 'yeah', 'yep'], 0:['no', 'nope', 'nah']}
allowed_conj = [',', ' but ', ' and ', ' ,', '.', ' .', '!', ' !']
qws = ['who', 'what', 'when', 'where', 'do', 'did', 'does', 'are', 'how']

##########################################################################################################################

def flatten(l):
	return [j for i in l for j in i]


def remove_ignored(inp):
	words_to_ignore = ['a', 'the']  # extend if needed
	return [i for i in inp if i not in words_to_ignore]


# handles potential short answers
def short_answer(q, a, yn):
	a = a.strip(" ").lower()
	correct_yn = [f"{r}{sep}" for r in yns[int(yn=='pos')] for sep in ['.',',','!',' ',';']]
	incorrc_yn = [f"{r}{sep}" for r in yns[int(yn=='neg')] for sep in ['.',',','!',' ',';']]
	if any([a.startswith(s) for s in incorrc_yn]): return 0
	
	starts_with_correct_yn = False
	for s in correct_yn:
		if a.startswith(s): 
			starts_with_correct_yn = True
			a = a[len(s):].strip()
			break			
	short_answer_map = json.load(open('util/short_answs.json'))
	qid = q.strip().lower().split()[:3]
	if qid[1] not in ['your', 'the'] or qid[:2]==['do', 'your']: qid=qid[:2]
	identifier = " ".join(qid)
	pos_short = [short_answer_map[identifier].lower()]
	neg_short = [f"{pos_short} not", f"{pos_short}n't"]
	sb, vb = pos_short[0].split()
	if vb in neg_contracts: neg_short += [f"{sb}{neg_contracts[vb]}not" if neg_contracts[vb].endswith(" ") else f"{sb} {neg_contracts[vb]}n't"]
	if vb in pos_contracts:	pos_short += [f"{sb}{pos_contracts[vb]}"]
	short_answers = {1:pos_short, 0:neg_short}

	correct_ans_starts = [(ans + conj) for ans in short_answers[int(yn=='pos')] for conj in allowed_conj]
	incorrc_ans_starts = [(ans + conj) for ans in short_answers[int(yn=='neg')] for conj in allowed_conj]
	if any([a.startswith(k) for k in correct_ans_starts]): return 1
	elif any([a.startswith(k) for k in incorrc_ans_starts]): return 0
	elif starts_with_correct_yn: return 3   # 3 : correct unless nli says wrong
	return -1 



# returns the token-level score
def check_rules(result):
	if result['Response'].lower().strip(" ") == "": return 0
	golds = [g.strip() for g in result['Reference'].lower().split('|')]
	if result['Type'] == 'yn':
		polarity = 'neg' if 'no' in golds else 'pos'
		sa_check = short_answer(result['Question'], result['Clean_Resp'], polarity)
		if sa_check != -1 : return sa_check	
	elif result['Type'] == 'wh':
		tokenizer = RegexpTokenizer(r"\w+")
		result = clean_response(result)
		digitized_resp = word_to_num(result['Clean_Resp']).lower()
		resp_words = tokenizer.tokenize(digitized_resp)
		gold_words = [tokenizer.tokenize(word_to_num(s)) for s in golds]

		gold_words = [remove_ignored(g) for g in gold_words]
		resp_words = remove_ignored(resp_words)
		if max([len(g) for g in gold_words]) <= 5 :
			return max([recall(set(g), set(resp_words)) for g in gold_words])
	return -1

	

# check if sentence is a question
def isq(x):
	if x.endswith('?'): return True
	if any([x.lower().startswith(qw) for qw in qws]): return True
	return False


# remove the interogative parts from agent's response
def clean_response(x):
	resp = x['Response']
	sents = [s.strip() for s in re.split(r"[;,.!]", resp)]
	if len(sents)>1 and isq(sents[-1]): resp = resp[:resp.find(sents[-1])]
	x['Clean_Resp'] = resp
	return x
