SHORT_PAUSE_TOKEN = "[short-pause]"
MEDIUM_PAUSE_TOKEN = "[medium-pause]"
LONG_PAUSE_TOKEN = "[long-pause]"
UNCLEAR_SPEECH_TOKEN = "[unclear-speech]"
LAUGHTER_TOKEN = "[laughter]"
NON_VERBAL_TOKEN = "[non-verbal]"
FILLER_UH_TOKEN = "[uh]"
FILLER_UM_TOKEN = "[um]"

SPECIAL_TOKENS: list[str] = [
    SHORT_PAUSE_TOKEN,
    MEDIUM_PAUSE_TOKEN,
    LONG_PAUSE_TOKEN,
    UNCLEAR_SPEECH_TOKEN,
    LAUGHTER_TOKEN,
    NON_VERBAL_TOKEN,
    FILLER_UH_TOKEN,
    FILLER_UM_TOKEN,
]

DISFLUENCY_TOKENS: list[str] = [
    *SPECIAL_TOKENS,
]

_TOKEN_INIT_DESCRIPTIONS: dict[str, str] = {
    SHORT_PAUSE_TOKEN: "a short silent pause",
    MEDIUM_PAUSE_TOKEN: "a medium silent pause",
    LONG_PAUSE_TOKEN: "a long silent pause",
    UNCLEAR_SPEECH_TOKEN: "unclear speech",
    LAUGHTER_TOKEN: "laughter",
    NON_VERBAL_TOKEN: "a non verbal vocal sound",
    FILLER_UH_TOKEN: "the filler uh",
    FILLER_UM_TOKEN: "the filler um",
}

def _description_vec(description: str, embeddings, tokenizer, old_vocab_size: int):
    import torch
    ids = [i for i in tokenizer.encode(description, add_special_tokens=False) if i < old_vocab_size]
    if not ids:
        return embeddings.weight[:old_vocab_size].mean(dim=0)
    return embeddings.weight[torch.tensor(ids, device=embeddings.weight.device)].mean(dim=0)


def register_tokens_with_init(model, processor, special_tokens: list[str]) -> list[int]:
    """Add special tokens and initialise each new embedding from its description mean.

    Both input and output (LM head) embeddings are set, so the model can meaningfully
    generate these token IDs even before fine-tuning.
    """
    import torch

    tokenizer = processor.tokenizer
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added == 0:
        return tokenizer.convert_tokens_to_ids(special_tokens)

    input_embeddings = model.get_input_embeddings()
    old_vocab_size = input_embeddings.weight.shape[0]

    model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    new_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    with torch.no_grad():
        for tok, new_id in zip(special_tokens, new_ids):
            desc = _TOKEN_INIT_DESCRIPTIONS.get(tok, tok.strip("[]").replace("-", " "))
            input_embeddings.weight[new_id] = _description_vec(
                desc, input_embeddings, tokenizer, old_vocab_size
            )
            if output_embeddings is not None:
                output_embeddings.weight[new_id] = _description_vec(
                    desc, output_embeddings, tokenizer, old_vocab_size
                )
    return new_ids
