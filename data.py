import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Mapping, Union

import os
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from transformers.data.data_collator import tolist, _torch_collate_batch

from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mask_ratio: float = 0.15
    decoder_mask_ratio: float = 0.15
    data_type: str = 'mixed'

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15) -> List[int]:
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            # Ignore special tokens
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]], mlm_prob: float) -> Dict[str, Any]:
        batch_input = self.tokenizer.pad(
            examples,
            padding='longest',
            max_length=self.max_seq_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens, mlm_probability=mlm_prob))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input['input_ids'], batch_mask)
        return {"input_ids": inputs, "attention_mask": batch_input['attention_mask'], "labels": labels}


    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.torch_call(examples, mlm_prob=self.encoder_mask_ratio)
        decoder_batch = self.torch_call(examples, mlm_prob=self.decoder_mask_ratio)
        batch['decoder_input_ids'] = decoder_batch['input_ids']
        batch['decoder_attention_mask'] = decoder_batch['attention_mask']
        batch['decoder_labels'] = decoder_batch['labels']
        return batch


class RetroMAEDataset(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args
        self.rng = random.Random()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]
