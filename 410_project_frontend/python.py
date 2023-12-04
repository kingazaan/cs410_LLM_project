import sys
input_from_js = str(sys.argv[1])

# import the dataset
import pandas as pd
import numpy as np
import os

# Directory path
#directory_path = 'C:\\Users\\azaan\\OneDrive\\Documents\\GitHub\\cs410_LLM_project\\data\\all_lectures.csv'
#Chris - Changed the directory path
directory_path = './all_lectures.csv'

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['Week Number', 'Lesson Number', 'Lesson Title', 'Transcript'])

# Read in csv to dataframe
df = pd.read_csv(directory_path)

# Display the resulting DataFrame
df.head()

# clean up words in dataset -- this includes removing stopwords
import regex as re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, brown

nltk.download("stopwords")
nltk.download("words")
nltk.download("brown")
#Chris - added following downloads
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# initialize dictionary
global_dictionary  = set(words.words()) | set(brown.words())
global_dictionary = {word.lower() for word in global_dictionary}
remove_words = list(stop_words) # might need to use word_tokenize
remove_words.extend(['Play', 'video', 'starting', 'at', '::', 'follow', 'transcript', 'natural', 'language', 'lecture', 'processing']) # add the common words that's include d in transcript

# Now start actually cleaning the text
def clean_text(text):
    text = text.lower() # lowercase
    text = text.replace('\n', ' ') # remove newline indicator
    text = re.sub(r'[^a-zA-Z\s]', '', text) # case
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+', '', text) # website
    text = re.sub(r'(\b\w+\b)(?: \1)+', r'\1', text) # remove duplicate next word after space
    text = re.sub(r'\b(?![aI]\b)\w\b', '', text)

    return text

# Remove stopwords and only keep words in dictionary
def remove_terms(text):
    text = clean_text(text)
    words = text.split()
    filtered_words = [word for word in words if word not in remove_words] # remove stopwords
    filtered_words = [word for word in filtered_words if word in global_dictionary] # remove if not in global dictionary
    return " ".join(filtered_words)

df['Transcript_Cleaned'] = df['Transcript'].apply(remove_terms)

df['Transcript_Cleaned'][0]

# Create bigrams and trigrams from data

# Function to filter bigrams or trigrams
def ngram_filter(ngram):
    tags = nltk.pos_tag(ngram)
    if not all(tag[1] in ['JJ', 'NN'] for tag in tags):
        return False
    if any(word in stop_words for word in ngram):
        return False
    if 'n' in ngram or 't' in ngram:
        return False
    if 'PRON' in ngram:
        return False
    return True

# Function to find top ngrams
def find_top_ngrams(texts, ngram_measures, min_freq=50, min_pmi=5, top_k=100):
    finder = nltk.collocations.BigramCollocationFinder.from_documents(texts)
    finder.apply_freq_filter(min_freq)
    ngram_scores = finder.score_ngrams(ngram_measures.pmi)
    filtered_ngrams = [ngram for ngram, pmi in ngram_scores if ngram_filter(ngram) and pmi > min_pmi]
    return [' '.join(ngram) for ngram in filtered_ngrams][:top_k]

bigram_measures = nltk.collocations.BigramAssocMeasures()
bigrams = find_top_ngrams([text.split() for text in df['Transcript_Cleaned']], bigram_measures)
trigram_measures = nltk.collocations.TrigramAssocMeasures()
trigrams = find_top_ngrams([text.split() for text in df['Transcript_Cleaned']], trigram_measures)

# Function to replace ngrams in text
def replace_ngrams(text):
    for gram in trigrams:
        text = text.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        text = text.replace(gram, '_'.join(gram.split()))
    return text

# Apply ngram replacements to the text
df['Grams'] = df['Transcript_Cleaned'].map(replace_ngrams)

# Tokenize reviews + remove stop words + filter only nouns
def tokenize_and_filter(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
    pos_comment = nltk.pos_tag(words)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    return filtered

df['Grams'] = df['Grams'].map(tokenize_and_filter)

# now I will make embeddings for my words, let's see if it works
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

results = set()
df['Grams'].apply(results.update)
vocab_size = len(results)

# Create a vocabulary dictionary
word_to_index = {word: idx for idx, word in enumerate(results)}

# Convert words to indices in your DataFrame
# AKA Encode these
# df['Grams_indices'] = df['Grams'].apply(lambda x: [word_to_index[word] for word in x])
def words_to_indices(words):
    return [word_to_index[word] for word in words]
df['Grams_indices'] = df['Grams'].apply(words_to_indices)

# Create a reverse dictionary
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Function to convert indices back to words
def indices_to_words(indices):
    return [index_to_word[idx] for idx in indices]

# # Apply the function to the 'Grams_indices' column
# Aka Decode these grams
# df['Decoded_Grams'] = df['Grams_indices'].apply(indices_to_words)

# Pad sequences to a specified length (e.g., maxlen)
maxlen = 60  # You can adjust this based on your data
padded_indices = pad_sequence([torch.LongTensor(seq) for seq in df['Grams_indices']], batch_first=True, padding_value=0)

# make a batch and set up parameters
block_size = 256
batch_size = 128
max_iters = 5000
learning_rate = 3e-4
eval_iters = 250
# new
n_embd = 128
n_layer = 4
dropout = 0.2
n_head = 4
prompt_size = 50

# Flatten the padded indices used to identify each word
data = flattened_indices = padded_indices.view(-1)
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]
# print(len(data))

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

x, y = get_batch('train')
#print(x)
#print(y)

# Estimating losses function
@torch.no_grad()

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Scaled dot product attention
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # create attention scores
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        drop = self.dropout(weights)
        
        # weighted aggregation of values
        v = self.value(x)
        out = drop @ v
        return out

# Multi-head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Creating a feedforward class
class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Creating a transformer block
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        y = self.attention(x)
        x = self.ln1(x + y)
        y = self.feedforward(x)
        x = self.ln2(x + y)
        return x

# Now to make a GPT model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Adding a positional embedding table as well
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) # added new parameter, n_embd
        # Adding 4 decoder layers
        self.blocks = nn.Sequential(*(Block(n_embd, n_head=n_head) for _ in range(n_layer)))
        # final layer normalization
        self.lm_f = nn.LayerNorm(n_embd)
        # unsure what this is below
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # std variables to help training converge better
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def forward(self, index, targets=None):
        B, T = index.shape

        # Add in token and posiional embeddings
        token_embd = self.token_embedding_table(index) # (B, T, C)
        pos_embd = self.positional_embedding_table(torch.arange(T)) # (T, C)
        x = token_embd + pos_embd # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.lm_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# model = GPTLanguageModel(vocab_size)

# in order to deal with a prompt, make the GPT model encounter a prompt size of around 50.
#model = GPTLangugeModel(vocab_size + prompt_size)
#Chris - changed spelling
model = GPTLanguageModel(vocab_size + prompt_size)

import pickle

# Save the model into a pickle file
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model, if necessary
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)

# Temporary cell 
model2 = GPTLanguageModel(vocab_size + prompt_size)

# index_to_word.update({0: ''})
# word_to_index[''] = word_to_index.pop('block')

# Add the prompt into the dictionary used for the training dataset
# index_to_word.update({0: ''})
# word_to_index[''] = word_to_index.pop('block')

# Issue a pompt, and we can try and generate an answer from it
#Chris - changed line below to line underneath it to use prompt from user
#prompt = 'Can you give me an overview on Probabilistic Latent Semantic Analysis'.split()
prompt = input_from_js.split()
# k = len(prompt)

# Find the maximum key in the existing dictionaries
max_key = max(word_to_index.values()) if word_to_index else -1

# Enumerate through the new words and add them to the dictionaries
for word in prompt:
    if word not in word_to_index:
        max_key += 1
        word_to_index[word] = max_key
        index_to_word[max_key] = word

# context = torch.zeros((1, 1), dtype=torch.long)
context = torch.tensor(words_to_indices(prompt), dtype=torch.long)
#Chris - replaced line below with the lines under it, this got rid of a KeyError
#generated_terms = indices_to_words(model2.generate(context.unsqueeze(0), max_new_tokens=50)[0].tolist())
words_generated = model2.generate(context.unsqueeze(0), max_new_tokens=50)[0].tolist()
words_generated = [x for x in words_generated if x <= len(index_to_word)]
generated_terms = indices_to_words(words_generated)
#print(' '.join(generated_terms[len(prompt):]))
#end of our code


data_to_pass_back = str(' '.join(generated_terms[len(prompt):]))
print(data_to_pass_back)

sys.stdout.flush()