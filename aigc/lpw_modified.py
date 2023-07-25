兼容berttokenizer
if isinstance(pipe.tokenizer, BertTokenizer):
    bos = pipe.tokenizer.cls_token_id
    eos = pipe.tokenizer.sep_token_id
    pad = 0
else:
    bos = getattr(pipe.tokenizer, "bos_token_id", 0)
    eos = getattr(pipe.tokenizer, "eos_token_id", 2)
    pad = getattr(pipe.tokenizer, "pad_token_id", eos)
