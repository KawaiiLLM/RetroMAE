#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from collections import defaultdict
from typing import Optional, Dict, Any

import datasets
from datasets import load_dataset

import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, Trainer,
)
from transformers.trainer_utils import is_main_process

from modeling import BertForTextCompression
from data import RetroMAEDataset, RetroMAECollator
from arguments import ModelArguments, DataTrainingArguments, CotMAEPreTrainingArguments as TrainingArguments
from datetime import datetime


class TrainerWithLogs(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inject Customised logging behavior
        self.customized_logging_list = defaultdict(list)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        A neat compute_loss that supports customized logging

        """
        outputs = model(**inputs)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Inject Customised logging behavior
        try:
            logs: dict = outputs.logs
        except:
            logs = None
        if logs is not None:
            for k, v in logs.items():
                # Set maxlen of list to avoid memory leak, useful when
                # customized_logging_list has not been cleaned correctly
                if len(self.customized_logging_list[k]) < 5000:
                    self.customized_logging_list[k].append(v)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # Inject Customised logging behavior
        for k, v in self.customized_logging_list.items():
            if len(v) > 0:
                logs[k] = round(sum(v) / len(v), 4)
        self.customized_logging_list.clear()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class RetroMAETrainer:
    def __init__(self,
                 data_args: DataTrainingArguments,
                 model_args: ModelArguments,
                 training_args: TrainingArguments):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def tokenize_function(self, examples: Dict[str, Any]):
        texts = ['[CLS] ' * (self.model_args.n_cls_tokens - 1) + text for text in examples['text']]
        tokenized_texts = self.tokenizer(texts, padding=False, truncation=True,
                                         max_length=self.data_args.max_seq_length)
        return {"input_ids": tokenized_texts['input_ids']}

    def train(self):
        data_file = os.path.join(self.data_args.train_path, 'train.jsonl')
        tokenized_data_file = os.path.join(self.data_args.train_path, 'tokenized_train_data')
        if os.path.exists(tokenized_data_file):
            train_set = datasets.load_from_disk(tokenized_data_file)
        else:
            train_set = load_dataset('json', data_files=data_file)['train']
            train_set = train_set.map(self.tokenize_function, remove_columns=train_set.column_names, batched=True,
                                      num_proc=3)
            train_set.save_to_disk(tokenized_data_file)
        train_set = RetroMAEDataset(train_set, self.data_args)
        eval_set = None
        data_collator = RetroMAECollator(
            tokenizer=self.tokenizer,
            max_seq_length=self.data_args.max_seq_length,
            encoder_mask_ratio=self.data_args.encoder_mask_ratio,
            decoder_mask_ratio=self.data_args.decoder_mask_ratio,
            data_type=self.data_args.data_type,
        )

        config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, cache_dir=self.model_args.cache_dir)
        model = BertForTextCompression.from_pretrained(
            pretrained_model_name_or_path=self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            n_cls_tokens=self.model_args.n_cls_tokens,
            use_decoder=self.model_args.use_decoder,
            n_decoder_layers=self.model_args.n_decoder_layers,
            enable_decoder_mlm=self.model_args.enable_decoder_mlm,
            decoder_mlm_coef=self.model_args.head_mlm_coef,
        )
        model.resize_token_embeddings(len(self.tokenizer))

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_set if self.training_args.do_train else None,
            eval_dataset=eval_set if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        if self.training_args.do_train:
            model_path = (
                self.model_args.model_name_or_path
                if (self.model_args.model_name_or_path is not None and os.path.isdir(
                    self.model_args.model_name_or_path))
                else None
            )
            trainer.train(model_path=model_path)
            trainer.save_model()  # Saves the tokenizer too for easy upload


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 1:
        model_args = ModelArguments(
            model_name_or_path='./models/bert-base-uncased',
            use_decoder=True,
            enable_decoder_mlm=True,
            n_decoder_layers=2,
            n_cls_tokens=1
        )
        data_args = DataTrainingArguments(
            max_seq_length=512,
            train_path='./data/wikipedia',
            encoder_mask_ratio=0.3,
            decoder_mask_ratio=0.5,
        )
        training_args = TrainingArguments(
            output_dir='./output/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            do_train=True,
            do_predict=False,
            learning_rate=1e-4,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            logging_steps=1000,
            logging_dir='./runs',
            save_steps=200000,
            save_total_limit=10,
            # max_steps=800000,
            num_train_epochs=1,
            fp16=True,
            warmup_ratio=0.1,
            gradient_accumulation_steps=1,
            overwrite_output_dir=True,
            dataloader_drop_last=True,
            dataloader_num_workers=4,
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Set seed before initializing model.
    set_seed(training_args.seed)

    retromae_trainer = RetroMAETrainer(data_args, model_args, training_args)
    retromae_trainer.train()


if __name__ == "__main__":
    main()
