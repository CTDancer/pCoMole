from transformers import EsmTokenizer

# Load the tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

print(f"Vocabulary size: {tokenizer.vocab_size}\n")

# Get all special token IDs
special_token_ids = []
special_tokens_info = {}

# Method 1: Extract from special_tokens_map (preferred for ESM)
if hasattr(tokenizer, "special_tokens_map") and tokenizer.special_tokens_map:
    for token_name, token_value in tokenizer.special_tokens_map.items():
        if isinstance(token_value, str):
            token_id = tokenizer.convert_tokens_to_ids(token_value)
            if isinstance(token_id, list) and len(token_id) > 0:
                token_id = token_id[0]
            if token_id is not None:
                special_token_ids.append(token_id)
                special_tokens_info[token_id] = (token_name, token_value)

# Method 2: Fallback to individual properties if needed
if len(special_token_ids) == 0:
    token_properties = [
        ("pad_token_id", "pad_token"),
        ("cls_token_id", "cls_token"),
        ("eos_token_id", "eos_token"),
        ("unk_token_id", "unk_token"),
        ("mask_token_id", "mask_token"),
        ("bos_token_id", "bos_token"),
        ("sep_token_id", "sep_token"),
    ]
    for prop_id, prop_name in token_properties:
        if hasattr(tokenizer, prop_id) and getattr(tokenizer, prop_id) is not None:
            token_id = getattr(tokenizer, prop_id)
            special_token_ids.append(token_id)
            token_value = getattr(tokenizer, prop_name, None)
            special_tokens_info[token_id] = (prop_name, token_value if token_value else f"<{prop_id}>")

special_token_ids = sorted(set(special_token_ids))

# Print special tokens
print("=" * 60)
print("SPECIAL TOKENS:")
print("=" * 60)
for token_id in special_token_ids:
    if token_id in special_tokens_info:
        name, value = special_tokens_info[token_id]
        print(f"  ID {token_id:3d}: {value!r:15s} ({name})")
    else:
        token_str = tokenizer.decode([token_id])
        print(f"  ID {token_id:3d}: {token_str!r:15s}")

# Get all non-special tokens
all_token_ids = set(range(tokenizer.vocab_size))
non_special_token_ids = sorted(all_token_ids - set(special_token_ids))

print(f"\n{'=' * 60}")
print(f"NON-SPECIAL TOKENS ({len(non_special_token_ids)} tokens):")
print("=" * 60)
print(f"  Token ID range: {non_special_token_ids[0]} to {non_special_token_ids[-1]}")

# Print first 20 and last 20 non-special tokens as examples
if len(non_special_token_ids) > 40:
    print("\n  First 20 non-special tokens:")
    for token_id in non_special_token_ids[:20]:
        token_str = tokenizer.decode([token_id])
        print(f"    ID {token_id:3d}: {token_str!r}")
    
    print("\n  Last 20 non-special tokens:")
    for token_id in non_special_token_ids[-20:]:
        token_str = tokenizer.decode([token_id])
        print(f"    ID {token_id:3d}: {token_str!r}")
else:
    # If few tokens, print all
    for token_id in non_special_token_ids:
        token_str = tokenizer.decode([token_id])
        print(f"    ID {token_id:3d}: {token_str!r}")

print(f"\n{'=' * 60}")
print(f"Summary:")
print(f"  Total vocabulary size: {tokenizer.vocab_size}")
print(f"  Special tokens: {len(special_token_ids)}")
print(f"  Non-special tokens: {len(non_special_token_ids)}")
print("=" * 60)