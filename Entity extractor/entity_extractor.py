#!/usr/bin/env python3

import json
from dandelion import DataTXT
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from os import environ


# Vars
original_text = 'data/policy.txt'
summarised_text = 'data/policy_summarised.txt'
entities_results = 'data/entities_policy.json'
output_results = 'data/results_policy.json'

# Tokenise sentences of summarised text
sentences = []
with open(summarised_text, 'r') as in_f:
    sentences = sent_tokenize(in_f.read())

# Entity Extraction through Dandelion APIs
token = environ['TOKEN']
datatxt = DataTXT(token=token)

entities = []
for s in sentences:
    entities.append(datatxt.nex(s, lang="en", include=(
        'alternate_labels', 'categories')))
with open(entities_results, 'w') as out_f:
    json.dump(entities, out_f)

# Get all labels for the extracted entities
data = []
# Read extracted entities from file
with open(summarised_text, 'r') as in_f:
    data = json.load(in_f)

labels = []
for e in data:
    for a in e['annotations']:
        ents = [e['entity'] for e in labels]
        if a['label'] not in ents:
            labels.append({'entity': a['label'], 'labels': [
                          a['spot'], a['label'], ] + a['alternateLabels']})

# Tokenise sentences of original text
sentences = []
with open(original_text, 'r') as in_f:
    sentences = sent_tokenize(in_f.read())

# Get all sentences which contain an entity, including its synonyms
s_per_labels = []
for e in labels:
    s_per_l = []
    for l in e['labels']:
        for s in sentences:
            if l.lower() in s.lower():  # If the sentence contains the labels
                words = word_tokenize(s)
                tags = pos_tag(words)
                for t in tags:
                    if t[1] == 'VB':    # If the sentence contains a verb
                        s_per_l.append(s)
    if len(s_per_l):
        s_per_labels.append(
            {'entity': e['entity'], 'sentences': list(dict.fromkeys(s_per_l))})
with open(output_results, 'w') as out_f:
    json.dump(s_per_labels, out_f)
