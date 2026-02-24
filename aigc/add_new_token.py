tokenizer = AutoTokenizer.from_pretrained(model_path)
policy_codes = load_policy_codes(policy_code_file)
policy_tokens = [f"<{code}>" for code in policy_codes]
print(f'before {len(tokenizer)}')
num_new_tokens = tokenizer.add_tokens(policy_tokens)
print(f'after {len(tokenizer)}')

print("新增token数量:", num_new_tokens)
model.model.resize_token_embeddings(len(tokenizer))

if num_new_tokens > 0:
    input_embeddings = model.model.get_input_embeddings().weight.data
    output_embeddings = model.model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True)

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
tokenizer.save_pretrained(new_model_dir)
model.model.save_pretrained(new_model_dir)
