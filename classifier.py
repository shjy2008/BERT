import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

import transformers

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

# import nltk
# # nltk.download('punkt_tab') # For POS(Part-of-Speech) Tagging
# # nltk.download('averaged_perceptron_tagger_eng') # For POS Tagging
# nltk.download('wordnet') # For WordNet (similar words)

# from nltk.corpus import wordnet as wn

# synsets = wn.synsets("pleasure")


import spacy
nlp = spacy.load("en_core_web_sm") # Load the pre-trained spaCy model

#### VLJ: For Task 1, implement functions inside the BertSentClassifier ####

TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = config

        # pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        if config.use_MSE_loss:
            num_output = 1 # Map to only 1 float number (0-1), can map to 0/1, or 0/1/2/3/4
        else:
            num_output = self.num_labels
        self.classifier_layer = torch.nn.Linear(config.hidden_size, num_output) 

    def forward(self, input_ids, attention_mask, POS_tag_ids, dep_tag_ids):
        # the final bert contextualize embedding is the hidden state of [CLS] token (the first token)
        bert_outputs = self.bert(input_ids, attention_mask, POS_tag_ids, dep_tag_ids)
        logits = self.classifier_layer(bert_outputs["pooler_output"])
        if self.config.use_MSE_loss:
            logits = F.sigmoid(logits) * (self.num_labels - 1) # <0.125:0  <0.375:1  <0.625:2  < 0.875:3  >0.875:4
        else:
            logits = F.log_softmax(logits, dim = 1)
        return logits


# create a custom Dataset Class to be used for the dataloader
class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.POS_tag_to_id = {"PAD": 0, "UNK": 1} # Part-of-Speech
        self.det_tag_to_id = {"PAD": 0, "UNK": 1} # Dependency parse tree

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele
    
    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        if self.p.POS_tag_enabled or self.p.dep_tag_enabled:
            # Get POS(part-of-speech) tag ids
            POS_tag_ids = []
            dep_tag_ids = []

            for sent_index, sent in enumerate(sents):
                # Split to words (this word count should be smaller than len(input_ids), because tokenizer split into sub-words)
                # words = nltk.word_tokenize(sent)
                # POS_tags = nltk.pos_tag(words)

                nlp_words = nlp(sent)
                words = [nlp_word.text for nlp_word in nlp_words]

                POS_tags = [nlp_word.pos_ for nlp_word in nlp_words] # Coarse-grained POS tag: .pos_  e.g. NOUN, VERB, ADJ, DET, PROPN
                # POS_tags = [nlp_word.tag_ for nlp_word in nlp_words] # Fine-grained POS tag: .tag_  e.g. NN, NNS, VB, VBD, VBG, JJ, JJR

                # Dependency Parse Tree
                dep_tags = [nlp_word.dep_ for nlp_word in nlp_words]

                # Get the tokens(sub-words) from tokenizer
                tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][sent_index])
                current_connected_tokens = ""
                word_index = 0
                sent_POS_tags = []
                sent_dep_tags = []

                # Get the POS tag for each token
                for token in tokens:
                    if token in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                        sent_POS_tags.append("PAD")
                        sent_dep_tags.append("PAD")
                    else:
                        cleaned_token = token.replace("##", "") # '##ing' -> 'ing'
                        current_connected_tokens += cleaned_token.lower()

                        POS_tag = None
                        dep_tag = None
                        assigned_tag = False
                        for i in range(word_index, len(words)):
                            word = words[i].lower()
                            if word.startswith(current_connected_tokens):
                                word_index = i
                                POS_tag = POS_tags[word_index]
                                dep_tag = dep_tags[word_index]
                                assigned_tag = True
                                if word == current_connected_tokens:
                                    word_index += 1
                                    current_connected_tokens = ""
                                break

                        if not assigned_tag:
                            POS_tag = "UNK"
                            dep_tag = "UNK"
                            current_connected_tokens = ""

                        sent_POS_tags.append(POS_tag)
                        sent_dep_tags.append(dep_tag)

                sent_POS_ids = []
                for POS_tag in sent_POS_tags:
                    if POS_tag not in self.POS_tag_to_id:
                        self.POS_tag_to_id[POS_tag] = len(self.POS_tag_to_id)
                    sent_POS_ids.append(self.POS_tag_to_id[POS_tag])
                POS_tag_ids.append(sent_POS_ids)

                sent_dep_ids = []
                for dep_tag in sent_dep_tags:
                    if dep_tag not in self.det_tag_to_id:
                        self.det_tag_to_id[dep_tag] = len(self.det_tag_to_id)
                    sent_dep_ids.append(self.det_tag_to_id[dep_tag])
                dep_tag_ids.append(sent_dep_ids)
            
            POS_tag_ids = torch.tensor(POS_tag_ids) if self.p.POS_tag_enabled else None
            dep_tag_ids = torch.tensor(dep_tag_ids) if self.p.dep_tag_enabled else None
        else:
            POS_tag_ids = None
            dep_tag_ids = None

        return token_ids, token_type_ids, attention_mask, labels, sents, POS_tag_ids, dep_tag_ids

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents, POS_tag_ids, dep_tag_ids = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'POS_tag_ids': POS_tag_ids,
                'dep_tag_ids': dep_tag_ids,
            })

        return batches


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, flag='train'):
    # specify the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = {}
    data = []

    with open(filename, 'r') as fp:
        for line in fp:
            label, org_sent = line.split(' ||| ')
            sent = org_sent.lower().strip()
            tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
            label = int(label.strip())
            if label not in num_labels:
                num_labels[label] = len(num_labels)
            data.append((sent, label, tokens))
    print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device, args):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents, b_POS_tag_ids, b_dep_tag_ids = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents'], \
                                                        batch[0]['POS_tag_ids'], batch[0]['dep_tag_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        if b_POS_tag_ids != None:
            b_POS_tag_ids = b_POS_tag_ids.to(device)
        if b_dep_tag_ids != None:
            b_dep_tag_ids = b_dep_tag_ids.to(device)

        logits = model(b_ids, b_mask, b_POS_tag_ids, b_dep_tag_ids)
        logits = logits.detach().cpu().numpy()
        if args.use_MSE_loss:
            preds = np.clip(np.round(logits).astype(int), 0, 4).flatten()
        else:
            preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = torch.device('cuda') if (args.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    best_dev_acc = 0

    # initialize the Senetence Classification Model
    if args.load_existing_model and os.path.exists(args.filepath):
        saved = torch.load(args.filepath, weights_only=False, map_location=device)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device, args)
        print("prev model dev accuracy: ", dev_acc)
        best_dev_acc = dev_acc
    else:
        #### Init model
        config = {'hidden_dropout_prob': args.hidden_dropout_prob,
                'num_labels': num_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'option': args.option,
                'use_MSE_loss': args.use_MSE_loss}

        config = SimpleNamespace(**config)
        model = BertSentClassifier(config)
        model = model.to(device)
        print(f"Train model from scratch. config: {config}")

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Scheduler with warmup
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    lr_scheduler = transformers.get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    ## run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents, b_POS_tag_ids, b_dep_tag_ids = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents'], batch[0]['POS_tag_ids'], batch[0]['dep_tag_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            if b_POS_tag_ids != None:
                b_POS_tag_ids = b_POS_tag_ids.to(device)
            if b_dep_tag_ids != None:
                b_dep_tag_ids = b_dep_tag_ids.to(device)

            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, b_POS_tag_ids, b_dep_tag_ids)
            if args.use_MSE_loss:
                loss = F.mse_loss(logits.view(-1), b_labels.view(-1).float())
            else:
                loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device, args)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device, args)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            test(args) # Test after saving the model with best result

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    if not os.path.exists(args.filepath):
        print(f"in test: filepath {args.filepath} not exists, return.")
        return
    with torch.no_grad():
        device = torch.device('cuda') if (args.use_gpu and torch.cuda.is_available()) else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False, map_location = device)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device, args)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device, args)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/sst-train.txt")
    parser.add_argument("--dev", type=str, default="data/sst-dev.txt")
    parser.add_argument("--test", type=str, default="data/sst-test.txt")
    parser.add_argument("--load_existing_model", type=int, default=1) # Load existing model or not, if 0, then train from scratch
    parser.add_argument("--do_training", type=int, default=1) # Do training or not, if 0, then not do training, do test directly
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--POS_tag_enabled", type=int, default=0)
    parser.add_argument("--dep_tag_enabled", type=int, default=0)
    parser.add_argument("--use_MSE_loss", type=int, default=0)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.filepath is None:
        #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
        args.filepath = f'sst-{args.option}-model.pt'
    seed_everything(args.seed)  # fix the seed for reproducibility

    if args.load_existing_model:
        print("---test before training---")
        test(args)

    print("---start training---")
    if args.do_training:
        train(args)
    print("---finish training---")

    test(args)
