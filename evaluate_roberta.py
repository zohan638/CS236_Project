import requests

def get_roberta_score(text):
	API_URL = "https://api-inference.huggingface.co/models/cointegrated/roberta-large-cola-krishna2020"
	headers = {"Authorization": f"Bearer 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"}

	def query(payload):
		response = requests.post(API_URL, headers=headers, json=payload)
		return response.json()
	
	outputs = query({
		"inputs": text,
	})
	
	scores = []
	for output in outputs:
		temp_dict = {output[0]['label'] : output[0]['score'],
							output[1]['label'] : output[1]['score']}
		scores.append((temp_dict['LABEL_0'], temp_dict['LABEL_1']))
	return scores

if __name__ == '__main__':
	text = "washington mentioned school from or confirmed use man the whose he cnn so up goal which shiite a stiles 62 government arrested pass victims stop more new the her site his anything is spokesman nation two week other victims . second of cnn have began . 100 sent in . met occurred public committee the that also got have and air side some were 15 we be are students in before accused children covered 1 took for between appeared provide john it rome the area after the i world will other illegal house john <unk> opposition al that her north even even officials from <unk> sailors every television is and than to drawers should <unk> match washington which two . still"
	text = 'This is a well written sentence. This is a poorly written sentence. This is a sentence that is neither well nor poorly written. This is a sentence that is both well and poorly written.'
	score = get_roberta_score(text)
	print("Score: ", score)