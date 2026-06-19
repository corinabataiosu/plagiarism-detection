import re
import hashlib

sample_text = "All the works of mortals are condemned to mortality; we live among things that are destined to perish."

print("Text original:")
print(f'"{sample_text}"\n')
print("-" * 60)

print("Baseline (Fereastra de 16 caractere)")
for i in range(min(10, len(sample_text) - 15)):
    chunk = sample_text[i:i+16]
    print(f"  Offset {i:02d}: '{chunk}'")

print("-" * 60)

print("N-grame de 4 cuvinte si hashing criptografic MD5")
words_iter = list(re.finditer(r'\b\w+\b', sample_text.lower()))
words = [(m.group(0), m.start()) for m in words_iter]
for i in range(len(words) - 3):
    ngram_phrase = " ".join([w[0] for w in words[i:i+4]])
    ngram_hash = hashlib.md5(ngram_phrase.encode('utf-8')).hexdigest()[:12]
    offset = words[i][1]
    print(f"  Offset {offset:02d}: '{ngram_phrase:<30}' -> Hash: {ngram_hash}")

print("-" * 60)

print("N-grame de 4 cuvinte si hashing FNV-1a")
def sim_fnv_word(s):
    h = 1469598103934665603
    for c in s:
        h ^= ord(c)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def sim_fnv_combine(h, x):
    h ^= x
    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

for i in range(len(words) - 3):
    ngram_words = [w[0] for w in words[i:i+4]]
    h = 1469598103934665603
    for w in ngram_words:
        h = sim_fnv_combine(h, sim_fnv_word(w))
    offset = words[i][1]
    print(f"  Offset {offset:02d}: {str(ngram_words):<40} -> uint64_t: {h}")