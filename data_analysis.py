"""
Prepare dataset
"""
import sys
import json
import logging
import random as rd

from datasets import load_dataset
from transformers import AutoTokenizer


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

LANG = "python"
MAX_SEQ_LEN = 512
MODEL_NAME = "EleutherAI/pythia-70m"
DATA_CFG = {
    "dataset": "bigcode/the-stack",
    "data_dir": "data/%s" % LANG,
}



if __name__ == '__main__':
    rng = rd.Random(123)
    ds = load_dataset(DATA_CFG["dataset"], data_dir=DATA_CFG["data_dir"], split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dev_data_count = int(len(ds) * 0.1)
    dev_data_ids = rng.sample(range(len(ds)), dev_data_count)
    train_model_inputs, dev_model_inputs = [], []

    for record_id, record in enumerate(ds):
        content = record["content"]
        # Remove license content (start with --)
        code_blocks = [code_block for code_block in content.split("\n\n") if not code_block.startswith("--")]
        code_blocks_input_ids = [tokenizer(code_block + "\n\n")["input_ids"] for code_block in code_blocks]

        j = 0
        cur_len = 0
        cur_input = []
        for i in range(len(code_blocks_input_ids)):
            if cur_len + len(code_blocks_input_ids[i]) > MAX_SEQ_LEN:
                # TODO(sanqiang): Execute code

                cur_model_input = {
                        "code_str": "".join([item for sublist in cur_input for item in sublist[1]]),
                        "code_input_ids": [item for sublist in cur_input for item in sublist[0]],
                        "origin_data_id": record_id
                }
                if record_id in dev_data_ids:
                    dev_model_inputs.append(cur_model_input)
                else:
                    train_model_inputs.append(cur_model_input)

            cur_len += len(code_blocks_input_ids[i])
            cur_input.append((code_blocks_input_ids[i], code_blocks[i]))

            while cur_len > MAX_SEQ_LEN:
                cur_len -= len(cur_input[0][0])
                del cur_input[0]

        logger.info("Progress %s" % round(record_id / len(ds), 2))

    train_output_dict = {
        "data": train_model_inputs,
        "version": 1.0
    }
    dev_output_dict = {
        "data": dev_model_inputs,
        "version": 1.0
    }
    json.dump(train_output_dict, open("./train.%s.seq512.json" % LANG, "w"), indent=4)
    json.dump(dev_output_dict, open("./dev.%s.seq512.json" % LANG, "w"), indent=4)

    logger.info("Generate %s training data" % len(train_model_inputs))
    logger.info("Generate %s dev data" % len(dev_model_inputs))




