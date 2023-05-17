import os
import sys
import logging

from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import set_seed, HfArgumentParser, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512
NUM_CPU = 16

os.environ["WANDB_PROJECT"] = "code_moonscript"


def train_model(model_args, training_args):
    dataset = load_dataset("json", data_files={
        "train": [model_args.input_train_path],
        "dev": [model_args.input_dev_path]
    }, field="data")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.pad_token = "<|padding|>"
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name)

    model_args.eft_mode = model_args.eft_mode.split(";")
    for model_arg in model_args.eft_mode:
        if model_arg.startswith("lora"):
            _, lora_r, lora_alpha = model_arg.split(":")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            logger.info("Use Peft: %s" % model_args.eft_mode)
            model.print_trainable_parameters()

    def preprocess_function(examples):
        encoder_inputs = tokenizer(examples["code_str"], max_length=MAX_SEQ_LEN, truncation=True, padding="max_length")
        examples["input_ids"] = encoder_inputs["input_ids"]
        examples["attention_mask"] = encoder_inputs["attention_mask"]
        examples["labels"] = examples["input_ids"]
        examples["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                              for labels in examples["labels"]]

        return examples

    dataset["train"] = dataset["train"].shuffle(123)
    encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names,
                                  num_proc=NUM_CPU, keep_in_memory=True)

    training_args.do_train = True
    training_args.do_eval = True
    training_args.load_best_model_at_end = True

    trainer = Trainer(
        model, training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"]
    )

    trainer.train()
    trainer.save_model()

@dataclass
class ModelArguments:
    model_name: str = field(
        default="EleutherAI/pythia-70m",
        metadata={"help": "Path to pretrain model or identifier from Huggingface."})

    input_train_path: str = field(
        default="./train.moonscript.seq512.json",
        metadata={"help": "Path to training dataset."})

    input_dev_path: str = field(
        default="./dev.moonscript.seq512.json",
        metadata={"help": "Path to dev dataset."})

    eft_mode: str = field(
        default="",
        metadata={"help": "Model mode for Efficient Fine tuning."})


if __name__ == '__main__':
    set_seed(123)
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    train_model(model_args, training_args)
