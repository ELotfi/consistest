from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")


american_number_system = {
	'zero': 0,
	'one': 1,
	'two': 2,
	'three': 3,
	'four': 4,
	'five': 5,
	'six': 6,
	'seven': 7,
	'eight': 8,
	'nine': 9,
	'ten': 10,
	'eleven': 11,
	'twelve': 12,
	'thirteen': 13,
	'fourteen': 14,
	'fifteen': 15,
	'sixteen': 16,
	'seventeen': 17,
	'eighteen': 18,
	'nineteen': 19,
	'twenty': 20,
	'thirty': 30,
	'forty': 40,
	'fifty': 50,
	'sixty': 60,
	'seventy': 70,
	'eighty': 80,
	'ninety': 90,
	'hundred': 100,
	'thousand': 1000,
	'million': 1000000,
	'billion': 1000000000,
	'point': '.'
}

decimal_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']




def number_formation(number_words):
	numbers = []
	for number_word in number_words:
		numbers.append(american_number_system[number_word])
	if len(numbers) == 4:
		return (numbers[0] * numbers[1]) + numbers[2] + numbers[3]
	elif len(numbers) == 3:
		return numbers[0] * numbers[1] + numbers[2]
	elif len(numbers) == 2:
		if 100 in numbers:
			return numbers[0] * numbers[1]
		else:
			return numbers[0] + numbers[1]
	else:
		return numbers[0]




def convert_w2n(clean_numbers):
	thousand_index = clean_numbers.index('thousand') if 'thousand' in clean_numbers else -1	
	total_sum = 0  # storing the number to be returned

	if len(clean_numbers) == 1:
			total_sum += american_number_system[clean_numbers[0]]
	else:
		if thousand_index > -1:
			thousand_multiplier = number_formation(clean_numbers[0:thousand_index])
			total_sum += thousand_multiplier * 1000

		if thousand_index > -1 and thousand_index != len(clean_numbers)-1:
			hundreds = number_formation(clean_numbers[thousand_index+1:])
		elif thousand_index == -1:
			hundreds = number_formation(clean_numbers)		
		else:
			hundreds = 0
		total_sum += hundreds	

	return total_sum





def word_to_num(sentence):
	number_sentence = sentence.replace('-', ' ').lower()
	if(number_sentence.isdigit()):  # return the number if user enters a number string
		return number_sentence

	number_sentence += ' END'
	split_words = tokenizer.tokenize(number_sentence)  # strip extra spaces and split sentence into words

	modified = []
	clean_numbers = []
	# removing and, & etc.
	in_number = False
	new_number = []
	counter = 0
	for i, word in enumerate(split_words):
		if word in american_number_system:
			new_number.append(word)
			if not in_number:		
				in_number = True
				modified.append(f'<NUM_{counter}>')
		elif word == 'and' and i>0 and split_words[i-1] in ['hundred', 'thousand']:
			pass
		else:
			modified.append(word)
			if in_number:
				clean_numbers.append(new_number)
				new_number = []
				in_number = False
				counter +=1

	if clean_numbers == []: return sentence

	for i, num in enumerate(clean_numbers):
		modified[modified.index(f'<NUM_{i}>')] = str(convert_w2n(num))
	
	return " ".join(modified[:-1])







