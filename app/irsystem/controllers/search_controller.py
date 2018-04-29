from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import os, json
from app.irsystem.models.database_helpers import *
from empath import Empath
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk import corpus as nltkCorp
import re
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import *
import time 
from bisect import bisect_left

project_name = "Fundy"
net_id = "Samantha Dimmer: sed87; James Cramer: jcc393; Dan Stoyell: dms524; Isabel Siergiej: is278; Joe McAllister: jlm493"

def process_donations(donations, issue):
	stemmer = PorterStemmer()
	words = set([stemmer.stem(w.lower()) for w in issue.split(" ")]) - set(nltkCorp.stopwords.words('english'))

	total = 0
	donations_list = []
	donations = list(donations)
	position_score = 0
	direct_matches = []

	for don in donations:
		don["org_data"] = get_org_data(don["DonorOrganization"])
		don["TransactionAmount"] = int(don["TransactionAmount"])
		don["org_data"]["donation_total"] = int(float(don["org_data"]["donation_total"]))
		donations_list.append(don)
		total += don["TransactionAmount"]

		found = False
		for word in words:
			if word in don["DonorOrganization"].lower():
				found = True
		if found:
			direct_matches.append(don)

		if float(don["org_data"]["democrat_total"])+float(don["org_data"]["republican_total"]) > 0:
			position_score += float(don["org_data"]["democrat_total"]) / (float(don["org_data"]["democrat_total"])+float(don["org_data"]["republican_total"]))

	if len(donations_list) > 0:
		position_score = round(position_score/len(donations_list)*100, 2)
	else:
		position_score = 50.00

	if len(direct_matches) <= 10:
		sample = sorted(direct_matches, key=lambda d:d["TransactionAmount"], reverse=True)
		sample += sorted(donations_list, key=lambda d:d["TransactionAmount"], reverse=True)[:min(len(donations_list), 10-len(sample))]
	else:
		sample = sorted(direct_matches[:10], key=lambda d:d["TransactionAmount"], reverse=True)

	return {
		"total": total,
		"sample": sample,
		"score": position_score,
	}


def get_issue_list(issue):
	stemmer = PorterStemmer()

	words = set([stemmer.stem(w.lower()) for w in issue.split(" ")]) - set(nltkCorp.stopwords.words('english'))
	synonyms = []
	for word in words:
		for synset in nltkCorp.wordnet.synsets(word):
			for lemma in synset.lemmas()[:min(5, len(synset.lemmas()))]:
				synonyms.append(str(lemma.name()))
	final = set(synonyms) | words

	return final

def binary_search(votes, politician):
	first = 0
	last = len(votes)-1
	found = False
	item = None
	politician_name = politician.split()
	politician_name_no_period = [a for a in politician_name if "." not in a][-1]

	while first<=last and not found:
		midpoint = (first + last)//2
		if votes[midpoint]["PoliticianName"] == politician:
			found = True
			item = votes[midpoint]
		else:
			name = votes[midpoint]["PoliticianName"].split()
			name_no_period = [a for a in name if "." not in a][-1]
			if politician_name_no_period < name_no_period:
				last = midpoint-1
			else:
				first = midpoint+1
	return item


# Calculate vote score based simply on if they voted yes or no on an issue
def vote_score_yes_no(votes):
	total_yes = 0
	total_no = 0
	if len(votes) > 0:
		for vote in votes:
			if vote["vote_position"] == "Yes":
				total_yes += 1
			elif vote["vote_position"] == "No":
				total_no += 1
	if total_yes > total_no:
		vote_score = 2.0*total_yes/(total_yes+total_no) - 1.0
	elif total_no > total_yes:
		vote_score = -2.0*total_no/(total_yes+total_no)
	else:
		vote_score = 0.0
	vote_score = round(vote_score, 2)
	return vote_score

# Returns percentage of people in same party this politician voted with
def vote_score_agree_with_party(votes, party):
	total_agree = 0.0
	total_party_votes = 0.0
	for vote in votes:
		if party == "R":
			for key in vote["republican"]:
				if key != "majority_position" and key != "not_voting":
					total_party_votes += vote["republican"][key]
					if key == vote["vote_position"].lower():
						total_agree += vote["republican"][key]
		elif party == "D":
			for key in vote["democratic"]:
				if key != "majority_position" and key != "not_voting":
					total_party_votes += vote["democratic"][key]
					if key == vote["vote_position"].lower():
						total_agree += vote["democratic"][key]
		if party == vote["party"]:
			total_party_votes -= 1
			total_agree -= 1
	if len(votes) == 0 or len(votes) == total_party_votes:
		score = 0
	else:
		score = round(total_agree/total_party_votes, 2)*100.0
	return score

def process_votes(raw_vote_data, query, politician, data):
	query_lower = query.lower()
	politician_party = "Unknown"
	for vote in raw_vote_data:
		issue_in_topics = False
		relevant_topic = ""
		if "subjects" in vote["vote"].keys():
			for topic in vote["vote"]["subjects"]:
				if query_lower in topic["name"].lower():
					issue_in_topics = True
					relevant_topic = topic["name"]
					break
		#If query and vote have similar topics or if query in bill description, add the vote to vote data
		# Add the title if there is a title, otherwise just add the description
		if "title" in vote["vote"]["bill"].keys() and vote["vote"]["bill"]["title"]:
			title = vote["vote"]["bill"]["title"]
			bill_title_relevant = query.lower() in vote["vote"]["bill"]["title"].lower()
		else:
			title = "N/A"
			bill_title_relevant = False
		if issue_in_topics or query.lower() in vote["vote"]["description"].lower() or bill_title_relevant:
			description = vote["vote"]["description"]
			url = vote["vote"]["url"]
			politician_vote = "Unknown"
			position = binary_search(vote["vote"]["positions"], politician)
			if position:
				politician_vote = position["vote_position"]
				if position["party"] == "R":
					politician_party = "Republican"
				elif position["party"] == "D":
					politician_party = "Democrat"
				else:
					politician_party = "Independent"
				democratic_votes = vote["vote"]["democratic"]
				republican_votes = vote["vote"]["republican"]
				independent_votes = vote["vote"]["independent"]
			if position and position["vote_position"] != "Not Voting" and position["vote_position"] != "Present" and position["vote_position"] != "Speaker" and politician_vote != "Unknown":
				data["votes"].append({"relevant_topic":relevant_topic, "url":url, "party":position["party"], "title":title, "description":description, "vote_position":politician_vote, "independent":independent_votes, "democratic":democratic_votes, "republican":republican_votes})
	#Do basic scoring system where score is % of time vote with party
	republican_vote_score = vote_score_agree_with_party(data["votes"], "R")
	democrat_vote_score = vote_score_agree_with_party(data["votes"], "D")
	data["vote_score_republican"] = republican_vote_score
	data["vote_score_democrat"] = democrat_vote_score
	data["party"] = politician_party
	if republican_vote_score == 0 and democrat_vote_score == 0:
		data["vote_scale"] = .5
	else:
		data["vote_scale"] = democrat_vote_score/(republican_vote_score+democrat_vote_score)
	return data
	
def tokenizer_custom(tweet):
    token = TweetTokenizer()
    stemmer = PorterStemmer()
    #remove links
    tweet = re.sub(r"http\S+", "", tweet)
    #remove user references
    tweet = re.sub(r"@\S+", "", tweet)
    #remove phone numbers
    tweet = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4})|(\(\d{3}\)\d{3}-\d{4}))', '', tweet)
    #remove punctuation
    tweet = re.sub(r'[^\w\s]','',tweet)
    #remove numbers
    tweet = re.sub(r"\d+", " ", tweet)
    #tokenize
    tokens = token.tokenize(tweet)
    #stem
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

#return (top n tweet indices, n top tweet scores)
def process_tweets(politician, query, n):
	tweets = get_tweets_by_politician(politician)
	vocab = json.load((open("app/irsystem/models/vocab.json", 'r')))['vocab']
	query_tokens = tokenizer_custom(query)

    #check query validity before proceeding
	valid_query = False
	for token in query_tokens:
		if token in vocab:
			valid_query = True
		else:
			query_tokens.remove(token)
	if valid_query == False:
		return ([],[])

	#dot query arrays
	query_dict = {}
	for token in query_tokens:
		postings = get_co_occurrence(token)['postings']
		for posting in postings:
			idx = posting['index']
			score = posting['score']
			word = vocab[idx]
			if word in query_dict:
				query_dict[word] *= score
			else:
				query_dict[word] = score

	#get similarity for each tweet
	sim_scores = []
	just_tweets = []
	for tweet in tweets:
		text = tweet['tweet_text']
		sentiment = tweet['sentiment']
		political = tweet["political"]
		favorites = tweet["favorites_count"]
		retweets = tweet["retweet_count"]
		just_tweets.append((text, sentiment, political, favorites, retweets))
		tokens = tweet["tokens"]
		#weight tweets that contain all words of query most highly
		sim_score = 0.0
		for token in tokens:
			if token in query_dict:
				sim_score += query_dict[token]
		sim_scores.append(sim_score)

	sim_scores = np.array(sim_scores)

	top_scores = -1*np.sort(-1*sim_scores)[:n]
	top_docs = np.argsort(-1*sim_scores)[:n]

	final_lst = []
	total_sentiment = 0.0
	for i in range(len(top_docs)):
		idx = top_docs[i]
		final_lst.append({"tweet": just_tweets[idx][0], "sentiment": just_tweets[idx][1], "score": top_scores[i], "political": just_tweets[idx][2],
			"favorites": just_tweets[idx][3], "retweets": just_tweets[idx][4]})
		total_sentiment += just_tweets[idx][1]["compound"]

	return (final_lst, total_sentiment)

@irsystem.route('/', methods=['GET'])
def search():
	politician_query = request.args.get('politician_name')
	free_form_query = request.args.get('free_form')
	data = None
	if not politician_query or not free_form_query: # no input
		output_message = 'Please provide an input'
		return render_template('search.html',
				name=project_name,
				netid=net_id,
				output_message=output_message,
				data=data,
		)
	else:
		output_message_politician = "Politician Name: " + politician_query 
		output_message_issue = "Issue: " + free_form_query
		data = {
			"politician": politician_query,
			"issue": free_form_query,
			"donations": [],
			"tweets": [],
			"votes": [],
			"vote_score": 0.0
		}
		if politician_query:	
			donation_data = get_relevant_donations(politician_query, get_issue_list(free_form_query))

			don_data = process_donations(donation_data, free_form_query)
			data["donations"] = don_data

			tweet_dict, total_sentiment = process_tweets(politician_query, free_form_query, 10)

			#return top 5 for now
			if len(tweet_dict) != 0:
				avg_sentiment = round(total_sentiment/10,2)
				data["tweets"] = {'tweet_dict': tweet_dict, 'avg_sentiment': avg_sentiment}

			t0 = time.time()

			raw_vote_data = get_votes_by_politician(politician_query)
			# Find all votes that have a subject that contains the issue typed in
			data = process_votes(raw_vote_data, free_form_query, politician_query, data)

			t1 = time.time()
			total = t1-t0
			print("TIMING: %d \n" % total)

		if free_form_query:
			pass
			#print("Need to implement this")
		return render_template('search.html',
				name=project_name,
				netid=net_id,
				output_message_politician=output_message_politician,
				output_message_issue=output_message_issue,
				data=data,
		)