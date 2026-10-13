import json

import einx
import numpy as np
import numpy.typing as npt
import torch
from tokenization import TOKENIZATION_MODELS_DIR, Tokenizer

from cs336_basics.training import load_checkpoint
from cs336_basics.training_loop import MODELS_DIR
from cs336_basics.transformer import SoftmaxLayer, TransformerLM

END_OF_TEXT = "<|endoftext|>"
SPECIAL_TOKENS = [END_OF_TEXT]


class Decoder:
    def __init__(self, config: dict, checkpoint_file):
        model_config = config["model"]
        load_checkpoint(checkpoint_file, None, None, self.config)
        self.softmax = SoftmaxLayer()

        self.model = TransformerLM(
            self.config["vocab_size"],
            self.config["context_length"],
            self.config["d_model"],
            self.config["num_layers"],
            self.config["num_heads"],
        )
        load_checkpoint(checkpoint_file, self.model, None)
        self.model.eval()
        self.tokenizer = Tokenizer.from_files(TOKENIZATION_MODELS_DIR / config["tokenizer_model_name"])
        self.end_of_text_id = self.tokenizer.encode(END_OF_TEXT)[0]

    @classmethod
    def from_model_id(cls, model_id):
        model_dir = MODELS_DIR / model_id
        config = json.loads((model_dir / "config").read_text())
        chkpt_file_path = model_dir / max(x for x in model_dir.iterdir() if "chkp" in x)
        return cls(config, chkpt_file_path)

    def _tokenize_prompt(self, prompt: str) -> npt.NDArray:
        return np.array(self.tokenizer.encode(prompt), dtype=np.uint16)

    def _get_token_id_from_logits(self, logits, temperature, top_p_thresh) -> int:
        logits = logits / temperature
        dist = self.softmax(logits).numpy()
        sort_indices = np.arange(len(dist))
        if top_p_thresh is not None:
            sort_indices = np.argsort(dist)[::-1]
            dist = dist[sort_indices]
            cum_dist = np.cumsum(dist)
            first_larger = np.argmax(cum_dist >= top_p_thresh)
            dist = dist[: first_larger + 1] / cum_dist[first_larger]
            sort_indices = sort_indices[: first_larger + 1]
        return np.random.choice(sort_indices, p=cum_dist)

    def complete(
        self,
        prompt: str,
        max_generated_tokens: int | None = None,
        temperature: float = 1.0,
        top_p_thresh: float | None = None,
    ) -> str:
        tokenized_prompt = self._tokenize_prompt(prompt)
        context = np.concat([tokenized_prompt, max_generated_tokens * [0]])
        context = einx.id("seq -> 1 seq", torch.tensor(context))
        input_len = len(context)
        num_out_tokens = 0
        for ind in range(input_len, input_len + max_generated_tokens):
            logits = self.model(context)[0, 0, :]
            next_token_id = self._get_token_id_from_logits(logits, temperature, top_p_thresh)
            if next_token_id == self.end_of_text_id:
                break
            num_out_tokens += 1
            context[0, ind] = next_token_id

        output_token_ids = context[input_len : input_len + num_out_tokens].numpy()
        return self.tokenizer.decode(output_token_ids)
