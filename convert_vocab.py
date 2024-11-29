import sys, ast

vocab_file = sys.argv[1]
vocab = None
with open(vocab_file, 'r') as f:
    vocab = f.readlines()

vocab_new = []
for line in vocab:
    parts = line.split(' ')
    assert len(parts) >= 3
    idx, token, token_len = int(parts[0]), ast.literal_eval(' '.join(parts[1:-1])), int(parts[-1])
    token = token.encode("utf-8") if isinstance(token, str) else token
    token_raw = "b'"
    for byte in token:
        token_raw += '\\x' + hex(byte)[2:].zfill(2)
    token_raw += "'"
    vocab_new.append(f"{idx} {token_raw} {token_len}\n")

with open("b_" + vocab_file, 'w') as f:
    f.writelines(vocab_new)