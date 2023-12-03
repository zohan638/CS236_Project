import requests

API_URL = "https://api-inference.huggingface.co/models/cointegrated/roberta-large-cola-krishna2020"
headers = {"Authorization": f"Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
text = "apple has also announced the allegations include the police from some places in connection or the latest other drug rights group in the same last"
output = query({
	"inputs": text,
})
score = output[0][0]['score']
print(score)