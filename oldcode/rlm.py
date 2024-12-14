
from typing import Set
import re
import numpy as np

from src.rules import Rule
from src.functions import levenshtein_distance, normalize_word

class ReinforcementLearning:
    """
    Reinforcement learning for stemming
    """

    def __init__(
        self,
        rules: list[Rule],
        corpus: Set[str],
        learning_rate: float = 0.1, # alpha
        learning_rate_decay: float = 0.995, # alpha decay
        min_learning_rate: float = 0.01, # minimum alpha
        discount_factor: float = 0.9, # gamma
        epsilon: float = 0.1, # epsilon
        epsilon_decay: float = 0.995, # epsilon decay
        min_epsilon: float = 0.01, # minimum epsilon
        curriculum_episodes: int = 100,
        max_step_per_episode: int = 100,
    ):
        """
        Initialize the reinforcement learning model.
        """
        self._rules = rules
        self._corpus = corpus
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._min_learning_rate = min_learning_rate
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._curriculum_episodes = curriculum_episodes
        self._episode_count = 0

        self._q_table = {}
        self._bag_of_words = []
        self._bag_of_results = []
        self._target_word = ''
        self._previous_min_distance = float('inf')
        self._previous_num_results = 0
        self._last_rules = []
        self._max_step_per_episode = max_step_per_episode

    @property
    def bag_of_words(self) -> list[str]:
        return self._bag_of_words
    
    @property
    def bag_of_results(self) -> list[str]:
        return self._bag_of_results
    
    @property
    def q_table(self) -> dict:
        return self._q_table

    def _min_distance(self) -> float:
        """
        Get the minimum distance from the bag of words and results
        """
        return min(levenshtein_distance(word, self._target_word) 
                   for word in self._bag_of_words + self._bag_of_results) if self._bag_of_words or self._bag_of_results else float('inf')
    
    def _get_pattern_success(self) -> float:
        """
        Get the pattern success rate
        """
        if not self._last_rules:
            return 0.0
        successful_applications = sum(1 for rule_idx in self._last_rules 
                                    if self._rules[rule_idx].pattern in str(self._bag_of_results))
        return successful_applications / len(self._last_rules)

    def _get_state(self) -> str:
        """
        Enhanced state representation incorporating more features
        and historical information.
        """
        # handle empty bag of words
        if not self._bag_of_words and not self._bag_of_results:
            return 'EMPTY|0|0|0|0|0'

        min_distance = self._min_distance()
        num_results = len(self._bag_of_results)
        num_rules_used = len(self._last_rules)

        # get morphological complexity score
        morphological_scores = []
        for word in self._bag_of_words:
            score = len(word)
            score += sum(1 for rule in self._rules if re.search(rule.pattern, word))
            morphological_scores.append(score)
        
        avg_complexity = (
            sum(morphological_scores) / len(morphological_scores) 
            if morphological_scores else 0
        )

        # pattern success rate
        pattern_success = self._get_pattern_success()
        rule_hash = hash(''.join(str(rule) for rule in self._last_rules[-5:])) % 1000 # limit size

        state = f"{num_results}|{num_rules_used}|{min_distance:.2f}|{avg_complexity:.2f}|{pattern_success:.2f}|{rule_hash}"
        return state
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current state
        Reward/penalized is based on:
        1. Word similarity to target from both bags
        2. Number of rules applied (fewer is better)
        3. Whether the result exists in corpus
        4. Penalize empty results or worse transformations
        """
        # base reward
        reward = 0

        # distance reward
        distance_improvement = self._previous_min_distance - self._min_distance()
        normalized_improvement = distance_improvement / len(self._target_word)
        reward += normalized_improvement * 2.0 # scale factor for distance component

        # reward if number of results is increasing
        if len(self._bag_of_results) > self._previous_num_results:
            reward += 1.0

        # big bonus if target word = result and bag_of_results = 1
        if self._target_word in self._bag_of_results and len(self._bag_of_results) == 1:
            reward += 5.0

        # extra reward if the target word is in the bag_of_results
        if self._target_word in self._bag_of_results and len(self._bag_of_results) > 1:
            reward += 3.0

        # penalize if number of rules used is increasing
        rule_penalty = -0.1 * len(self._last_rules)
        rule_penalty *= (1 + self._episode_count / self._curriculum_episodes) # scale with progress
        reward += rule_penalty

        # penalize excessive transformations
        if len(self._bag_of_words) > len(self._target_word) * 3:
            reward -= 0.5

        # update previous state
        self._previous_min_distance = self._min_distance()
        self._previous_num_results = len(self._bag_of_results)

        return reward
    
    def _choose_action(self) -> int:
        """
        Choose an action based on the Q-values
        """
        # get available actions (not in used rules)
        used_rules_indices = set(self._last_rules)
        available_actions = [
            i for i in range(len(self._rules)) 
            if i not in used_rules_indices
        ]

        # if all rules used, return -1
        if not available_actions:
            return -1
        
        # epsilon-greedy selection
        if np.random.random() < self._epsilon:
            return np.random.choice(available_actions)
        else:
            # exploitation by reading Q-table
            current_state = self._get_state()
            if current_state not in self._q_table:
                self._q_table[current_state] = {i: 0.0 for i in range(len(self._rules))}
            
            # get Q-values for available actions
            q_values = self._q_table[current_state]
            available_q_values = {i: q_values[i] for i in available_actions}

            # return action with highest Q-value
            return max(available_q_values.items(), key=lambda x: x[1])[0]


    def _update_q_table(self, current_state: str, action: int, reward: float, new_state: str) -> None:
        """
        Update the Q-table and return terminal state
        """
        if current_state not in self._q_table:
            self._q_table[current_state] = {i: 0.0 for i in range(len(self._rules))}
        if new_state not in self._q_table:
            self._q_table[new_state] = {i: 0.0 for i in range(len(self._rules))}

        current_q = self._q_table[current_state][action]

        # if terminal state, don't include future rewards
        is_terminal = self._target_word in self._bag_of_results
        next_max_q = 0 if is_terminal else max(self._q_table[new_state].values())

        # Q-learning update formula
        new_q = current_q + self._learning_rate * (
            reward + 
            (0 if is_terminal else self._discount_factor * next_max_q) - current_q
        )

        self._q_table[current_state][action] = new_q

        # Terminal condition if target reached
        return is_terminal
    
    def _apply_rule(self, rule_idx: int) -> None:
        """
        Apply the rule to the bag of words and update the bag of results
        """
        rule = self._rules[rule_idx]
        words_to_add = []
        results_to_add = []

        for word in self._bag_of_words:
            if re.match(rule.pattern, word, re.UNICODE): # if word matches rule pattern
                if len(rule.replacements) == 0:
                    new_word = re.sub(rule.pattern, rule.replacement, word)
                    words_to_add.append(new_word)

                    # add to results if in corpus
                    if new_word in self._corpus:
                        results_to_add.append(new_word)
                else:
                    # multiple replacements
                    for replacement in rule.replacements:
                        new_word = re.sub(rule.pattern, replacement, word)
                        words_to_add.append(new_word)

                        # add to results if in corpus
                        if new_word in self._corpus:
                            results_to_add.append(new_word)

        # apply changes
        if words_to_add:
            self._bag_of_words.extend(words_to_add)
            # remove duplicates
            self._bag_of_words = list(dict.fromkeys(self._bag_of_words))

        if results_to_add:
            self._bag_of_results.extend(results_to_add)
            # remove duplicates
            self._bag_of_results = list(dict.fromkeys(self._bag_of_results))

        self._last_rules.append(rule_idx)
        

    def train(self, word: str, target_word: str) -> None:
        """
        Train the model using Q-learning to find optimal rule sequences for word stemming.
        
        The training process:
        1. Runs multiple episodes (defined by curriculum_episodes)
        2. In each episode:
           - Applies rules based on epsilon-greedy strategy
           - Updates Q-values based on rewards
           - Continues until target word is found or no more rules can be applied
        3. Progressively reduces exploration (epsilon) and learning rate
        
        Args:
            word (str): The input word to be stemmed
            target_word (str): The desired stemmed form of the word
            
        Note:
            - The model learns Q-values for state-action pairs
            - States are encoded as: "num_results|num_rules_used|min_distance|avg_complexity"
            - Actions correspond to applying specific stemming rules
            - Training affects the internal Q-table used for predictions
        """
        # normalize input word and target word
        self._bag_of_words = [normalize_word(word)]
        self._target_word = normalize_word(target_word)

        # train for multiple episodes
        for episode in range(self._curriculum_episodes):
            # decay epsilon
            self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
            # decay learning rate
            self._learning_rate = max(self._min_learning_rate, self._learning_rate * self._learning_rate_decay)

            # reset episode state
            self._bag_of_results = []
            self._last_rules = []
            self._previous_min_distance = float('inf')
            self._previous_num_results = 0
            self._episode_count += 1

            # run episode until terminal state
            step_count = 0
            while step_count < self._max_step_per_episode:
                # get current state and choose action
                current_state = self._get_state()
                action = self._choose_action()

                if action == -1:
                    break
                
                # apply rule
                self._apply_rule(action)
                new_state = self._get_state()
                reward = self._calculate_reward()

                # check if terminal state
                terminal = self._update_q_table(current_state, action, reward, new_state)

                if terminal:
                    break
                
                step_count += 1

    def predict(self, word: str) -> str:
        """
        Predict the stem of a word based on the trained model.
        Uses the Q-table to determine the optimal sequence of rules to apply.
        Returns the first valid stemmed word found in the corpus.
        
        Args:
            word (str): Input word to be stemmed
            
        Returns:
            str: Stemmed word if found in corpus, otherwise original word
        """
        # Normalize input word
        word = normalize_word(word)
        
        # Initialize state
        self._bag_of_words = [word]
        self._bag_of_results = []
        self._last_rules = []
        self._previous_min_distance = float('inf')
        self._previous_num_results = 0
        
        # Track best result and its score
        best_result = word
        best_score = float('-inf')
        
        max_steps = len(self._rules)
        step_count = 0
        
        # while step_count < max_steps:
        #     current_state = self._get_state()
            
        #     # Get Q-values for current state
        #     if current_state not in self._q_table:
        #         # If state not in Q-table, initialize it
        #         self._q_table[current_state] = {i: 0.0 for i in range(len(self._rules))}
            
        #     # Get available actions (excluding already used rules)
        #     used_rules_indices = set(self._last_rules)
        #     available_actions = [
        #         i for i in range(len(self._rules)) 
        #         if i not in used_rules_indices
        #     ]
            
        #     if not available_actions:
        #         break
                
        #     # Get Q-values for available actions
        #     q_values = self._q_table[current_state]
        #     available_q_values = {i: q_values[i] for i in available_actions}
            
        #     # Choose action with highest Q-value
        #     best_action = max(available_q_values.items(), key=lambda x: x[1])[0]
            
        #     # Apply the chosen rule
        #     prev_results_len = len(self._bag_of_results)
        #     self._apply_rule(best_action)
            
        #     # Calculate score for current state
        #     current_score = self._calculate_score()
            
        #     # Update best result if current state is better
        #     if current_score > best_score and self._bag_of_results:
        #         best_score = current_score
        #         best_result = min(self._bag_of_results,
        #                         key=lambda x: levenshtein_distance(x, word))
            
        #     # Stop if we found results and no improvement in last step
        #     if self._bag_of_results and len(self._bag_of_results) == prev_results_len:
        #         break
                
        #     step_count += 1
        
        # return best_result

    
        


        
        
