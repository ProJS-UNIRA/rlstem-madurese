import re
from src.rl2 import ReinforcementLearning as RL
from src.rules import rules
from src.functions import get_unicode_code, normalize_word


corpus = open("data/lemma_expanded.txt").read().splitlines()
for i, word in enumerate(corpus):
    corpus[i] = normalize_word(word)

words = [
    ("èsèkot", "sèkot"),
    ("ètabâng", "tabâng"),
    ("abhâlik", "bhâlik"),
    ("aḍhârât", "ḍhârât"),
    ("nabâraghi", "tabâr"),
    ("aghellâng", "ghellâng"),
    ("kawâjibhan", "wâjib"),
    ("nako'e", "tako'"),
    ("èjungkataghi", "jungkat"),
    ("ètalèè", "talè"),
    ("katorodhan", "torot"),
    ("tabhâlighâ", "bhâlik"),
    ("pangaleman", "alem"),
    ("epangala'", "kala'")
]

optimizer = RL(
    rules=rules,
    num_words=len(words)
)

optimizer.train(words, episodes=10)



