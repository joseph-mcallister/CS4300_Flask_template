import json
import gzip
import indicoio

indicoio.config.api_key = 'ca750e4a8184087bf1d6c3892b162e6b'

with gzip.GzipFile("tweet_data_zip", 'r') as data_in:
	data_bytes = data_in.read()

data_str = data_bytes.decode('utf-8')
tweet_data = json.loads(data_str)

just_tweets = [tweet["tweet_text"] for tweet in tweet_data]




# # single example
# indicoio.political("I have a constitutional right to bear arms!")

max_send = 30000

# batch example
finished = False
i = 12
while not finished:
	up_bound = min((i+1)*max_send, len(just_tweets))
	if up_bound == len(just_tweets):
		finished = True
	to_send = just_tweets[i*max_send:up_bound]
	stuff = indicoio.political(to_send)
	print(len(stuff))
	stuff2 = {"stuff": stuff}
	print ("dumping", i)
	json.dump(stuff2,(open("testing5.json", 'a')))
	print ("done dumping", i)
	i += 1


