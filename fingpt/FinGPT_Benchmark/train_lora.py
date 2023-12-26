"""
This script trains a LoRA (Low-Rank Attention) model using the FinGPT architecture.
It loads a pre-trained causal language model, tokenizes the training and testing datasets,
and fine-tunes the model using the specified training arguments.
The trained model is then saved for future use.
"""

import os
import sys
import argparse
from datetime import datetime
from functools import partial
import datasets
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.integrations import TensorBoardCallback
# Importing LoRA specific modules
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
from utils import *


# Replace with your own api_key and project name
os.environ['WANDB_API_KEY'] = 'ecf1e5e4f47441d46822d38a3249d62e8fc94db4'
os.environ['WANDB_PROJECT'] = 'fingpt-benchmark'


def main(args):
    """
    Main function to execute the training script.

    :param args: Command line arguments
    """

    # Parse the model name and determine if it should be fetched from a remote source
    model_name = parse_model_name(args.base_model, args.from_remote)
    
    # Load the pre-trained causal language model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        # device_map="auto",
        trust_remote_code=True
    )
    # Print model architecture for the first process in distributed training
    if args.local_rank == 0:
        print(model)

    # Load tokenizer associated with the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apply model specific tokenization settings
    if args.base_model != 'mpt':
        tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('
