import nltk

nltk.download('punkt_tab') # For POS(Part-of-Speech) Tagging
nltk.download('averaged_perceptron_tagger_eng') # For POS Tagging
nltk.download('wordnet') # For WordNet (similar words)

from nltk.corpus import wordnet as wn
import random

def get_wordnet_POS(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def get_synonyms(word, POS_tag = None):
    synonyms = []
    synsets = wn.synsets(word, pos = POS_tag)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)
    return synonyms

# synsets = wn.synsets("pleasure")
# print(get_synonyms("touching"))

    
def get_new_sentences(sent):
    tokens = nltk.tokenize.word_tokenize(sent.lower())
    tagged_tokens = nltk.tag.pos_tag(tokens)

    candidate_indices = []
    index_to_POS_dict = {}

    for i, (word, tag) in enumerate(tagged_tokens):
        wordnet_POS = get_wordnet_POS(tag)
        if wordnet_POS in [wn.ADJ, wn.ADV] or (wordnet_POS == wn.VERB and len(word) > 3): # exclude verbs like 'am', 'is', 'was', 'do'
            candidate_indices.append(i)
            index_to_POS_dict[i] = wordnet_POS

    if len(candidate_indices) == 0:
        return []

    new_sentences = []
    for _ in range(2):
        replace_count = len(candidate_indices) // 2 + 1 # replace half the candidates + 1
        replace_indices = random.sample(candidate_indices, replace_count)
        # new_tokens = tokens.copy()

        is_new = False
        for index in replace_indices:
            synonyms = get_synonyms(tokens[index].lower(), index_to_POS_dict[index])
            if synonyms:
                new_word = random.choice(synonyms)
                tokens[index] = new_word
                is_new = True
        
        if is_new:
            new_sent = ' '.join(tokens)
            new_sentences.append(new_sent)

    return new_sentences

# sentence = "The cinematic equivalent of patronizing a bar favored by pretentious , untalented artistes who enjoy moaning about their cruel fate ."

# print (get_new_sentences(sentence))


read_file = './data/sst-train.txt' # Source: https://amazon-reviews-2023.github.io/ Movies_and_TV
write_file = './data_ext/sst-train-ext2.txt'

count = 0
with open(read_file, "r") as f_dataset:
    with open(write_file, 'w') as f_new:
        while True:
            line = f_dataset.readline()
            if not line:
                break
            
            # count += 1
            # if count > 10:
            #     break

            label, org_sent = line.split(' ||| ')
            sent = org_sent.lower().strip()

            new_sents = get_new_sentences(sent)

            for new_sent in new_sents:
                f_new.write(f'{label} ||| {new_sent}\n')
