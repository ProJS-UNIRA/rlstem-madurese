import re
from lib.rl import ReinforcementLearning as RL
from lib.rules import rules
from lib.functions import normalize_word, get_unicode_code

corpus = open("data/lemma_expanded.txt").read().splitlines()
for i, word in enumerate(corpus):
    corpus[i] = normalize_word(word)

rl = RL(
    rules=rules,
    corpus=corpus,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1,
    curriculum_episodes=500,
)

# rl.train("èsèkot", "sèkot")
# rl.train("epangala'", "kala'")
rl.train("pangaleman", "alem")
# rl.train("sakejjhâ'", "kejjhâ'")

print(rl.bag_of_words)
print(rl.bag_of_results)


