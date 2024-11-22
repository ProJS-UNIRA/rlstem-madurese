import re
from src.rlm import ReinforcementLearning as RL
from src.rules import rules
from src.functions import normalize_word

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

rl.train("èsèkot", "sèkot")
# rl.train("epangala'", "kala'")
# rl.train("pangaleman", "alem")
# rl.train("sakejjhâ'", "kejjhâ'")

print(rl.predict("èkala'"))
print(rl.predict("èsèkot"))
print(rl.predict("ecampor"))
# print(rl.bag_of_words)
# print(rl.bag_of_results)


