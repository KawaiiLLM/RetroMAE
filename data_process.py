import json
import random

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def create_wiki_data(tokenizer_name: str,
                     max_seq_length: int,
                     short_seq_prob: float,
                     output_dir: str):
    import nltk
    nltk.download('punkt')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    target_length = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    def tokenize_sentences(examples):
        sentences_ids = []
        for sentences in examples['sentences']:
            sentences_ids.append(
                tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                          return_token_type_ids=False)['input_ids'])
        return {"sentences_ids": sentences_ids}

    def split_sentences(examples):
        sentences = nltk.sent_tokenize(examples["text"])
        return {"sentences": sentences}

    def sentences_to_sample(examples):
        text = []
        for sents in examples['sentences_ids']:
            curr_input_ids = []
            curr_tgt_len = target_length if random.random() > short_seq_prob else random.randint(3, target_length)
            for sent in sents:
                if len(curr_input_ids) >= curr_tgt_len:
                    text.append(tokenizer.decode(curr_input_ids))
                    curr_input_ids = []
                    curr_tgt_len = target_length if random.random() > short_seq_prob \
                        else random.randint(3, target_length)
                curr_input_ids.extend(sent)
            if len(curr_input_ids) > 0:
                text.append(tokenizer.decode(curr_input_ids))
        return {'text': text}

    dataset = load_dataset("wikipedia", "20220301.en", cache_dir='./data')['train']
    # dataset = datasets.Dataset.from_dict(dataset[:100])
    dataset = dataset.map(split_sentences, num_proc=4, remove_columns=["id", "url", "title", "text"])
    dataset = dataset.map(tokenize_sentences, num_proc=4, batched=True, remove_columns=["sentences"])
    dataset = dataset.map(sentences_to_sample, num_proc=4, batched=True, remove_columns=["sentences_ids"])
    with open(output_dir, 'w', encoding='utf-8') as of:
        idx = 0
        for sample in tqdm(dataset):
            of.write(json.dumps({"id": idx, "text": sample['text']}, ensure_ascii=False) + '\n')
            idx += 1


if __name__ == '__main__':
    create_wiki_data(tokenizer_name='./models/bert-base-uncased', max_seq_length=512, short_seq_prob=1.0,
                     output_dir='./data/wikipedia/train.jsonl')

