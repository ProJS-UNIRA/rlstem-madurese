"""
Module for reinforcement learning for stemming.
"""

from collections import deque
import random
import re
from typing import Set

import numpy as np

from src.rules import Rule
from src.functions import normalize_word, levenshtein_distance

class ReinforcementLearning:
    """
    Reinforcement learning for stemming
    """

    def __init__(
        self,
        rules: list[Rule],
        corpus: Set[str],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        curriculum_episodes: int = 100,
    ):
        """
        Initialize the reinforcement learning model.
        """
        self._rules = rules
        self._corpus = corpus
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._epsilon = epsilon

        self._q_table = {}
        self._bag_of_words = []
        self._bag_of_results = []
        self._target_word = ''
        self._last_rules = [] # containing indexes of rules in self._rules
        self._previous_min_distance = float('inf')
        self._episode_count = 0
        self._curriculum_episodes = curriculum_episodes

        self._temperature = 1.0 # initial temperature
        self._min_temperature = 0.1 # minimum temperature
        self._temperature_decay = 0.995 # temperature decay rate
        self._action_counts = {} # track action usage

        # self._experience_buffer = deque(maxlen=1000) # store experience tuples
        # self._batch_size = 32 # batch size for experience replay
        # self._min_experience_size = 100 # minimum experience size for training
        # self._success_pattern = {} # track success rule sequences

    @property
    def bag_of_words(self) -> list[str]:
        return self._bag_of_words
    
    @property
    def bag_of_results(self) -> list[str]:
        return self._bag_of_results
    
    
    def _get_state(self) -> str:
        """
        Get the current state of the bag of words.
        Return a string that uniquely identifies the current state, containing:
        - current bag of words
        - current bag of results
        - applied rules (to avoid repeating same rule)
        """
        min_distance = min(levenshtein_distance(word, self._target_word) 
                      for word in self._bag_of_results + self._bag_of_words) if self._bag_of_results or self._bag_of_words else float('inf')
        
        # calculate remaining available rules
        used_rules = set(self._last_rules)
        available_rules_count = len(self._rules) - len(used_rules)

        # get morphological complexity score
        morphological_scores = []
        for word in self._bag_of_words + self._bag_of_results:
            score = len(word)
            score += sum(1 for rule in self._rules if re.search(rule.pattern, word))
            morphological_scores.append(score)
        avg_complexity = sum(morphological_scores) / len(morphological_scores) if morphological_scores else 0

        applied_rules = '_'.join(str(rule) for rule in self._last_rules)
        state = f"{self._target_word}|{min_distance:.2f}|{available_rules_count}|{avg_complexity:.2f}|{applied_rules}"
        return state
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current state.
        Implement curriculum learning based on episode progress.
        Reward/penalized is based on:
        1. Word similarity to target from both bags
        2. Number of rules applied (fewer is better)
        3. Whether the result exists in corpus
        4. Penalize empty results or worse transformations
        """

        # base reward
        reward = 0

        # get curremt distance from both bags
        distances = [levenshtein_distance(word, self._target_word)
                     for word in self._bag_of_results + self._bag_of_words]
        current_min_distance = min(distances) if distances else float('inf')

        # Normalize distance improvement to [-1, 1] range
        max_possible_improvement = len(self._target_word)  # maximum possible improvement
        distance_improvement = self._previous_min_distance - current_min_distance
        normalized_improvement = distance_improvement / max_possible_improvement
        reward += normalized_improvement * 2.0  # Scale factor for distance component
        
        # 2. Efficiency reward: penalize unnecessary transformations
        unique_words = len(set(self._bag_of_words + self._bag_of_results))
        efficiency_penalty = -0.1 * (unique_words / len(self._target_word))
        reward += efficiency_penalty
        
        # 3. Corpus matching reward with word similarity consideration
        for word in self._bag_of_results:
            if word in self._corpus:
                # Base corpus reward
                corpus_reward = 0.5
                
                # Additional reward based on similarity to target
                similarity = 1 - (levenshtein_distance(word, self._target_word) / 
                                max(len(word), len(self._target_word)))
                corpus_reward *= (1 + similarity)
                
                # Extra reward for exact match
                if word == self._target_word:
                    corpus_reward *= 2.0
                    
                reward += corpus_reward
        
        # 4. Rule usage efficiency
        rule_penalty = -0.1 * len(self._last_rules)
        rule_penalty *= (1 + self._episode_count / self._curriculum_episodes)  # Scale with progress
        reward += rule_penalty
        
        # 5. Progress-based curriculum scaling
        progress_ratio = self._episode_count / self._curriculum_episodes
        curriculum_factor = 1 + progress_ratio
        reward *= curriculum_factor
        
        # Update previous min distance for next iteration
        self._previous_min_distance = current_min_distance
        
        return reward
      
    def _softmax(self, q_values: dict) -> dict:
        """
        Compute softmax probabilities for Q-values.
        """
        values = np.array(list(q_values.values()), dtype=np.float64)
        
        # Replace any NaN/inf values with 0
        values = np.nan_to_num(values, nan=0.0, posinf=500, neginf=-500)
        
        # handle potential extreme values
        values = np.clip(values, -500, 500)
        
        # numerical stability
        max_val = np.max(values)
        shifted_values = values - max_val
        exp_values = np.exp(shifted_values / max(self._temperature, 1e-10))
        
        # avoid division by zero and check for NaN
        sum_exp_values = np.sum(exp_values)
        if sum_exp_values == 0 or np.isnan(sum_exp_values):
            probabilities = np.ones_like(exp_values) / len(exp_values)
        else:
            probabilities = exp_values / sum_exp_values
            # Final NaN check
            if np.any(np.isnan(probabilities)):
                probabilities = np.ones_like(exp_values) / len(exp_values)
        
        return dict(zip(q_values.keys(), probabilities))

    def _choose_action(self) -> int:
        """
        Enhanced action selection with softmax exploration and action history.
        """
        current_state = self._get_state()

        # Initialize Q-table for the current state if not exists
        if current_state not in self._q_table:
            self._q_table[current_state] = {
                i: 0.0 for i in range(len(self._rules))
            }
        if current_state not in self._action_counts:
            self._action_counts[current_state] = {
                i: 0 for i in range(len(self._rules))
            }

        # Get available actions (not in used rules)
        used_rules_indices = set(self._last_rules)
        available_actions = [
            i for i in range(len(self._rules)) 
            if i not in used_rules_indices
        ]

        # if all rules used, return -1
        if not available_actions:
            return -1

        # Calculate UCB scores for available actions
        ucb_scores = {}
        total_visits = sum(self._action_counts[current_state].values()) + 1
        c = 1.0  # exploration constant
        
        for action in available_actions:
            q_value = self._q_table[current_state][action]
            visit_count = self._action_counts[current_state][action]
            
            # UCB formula with additional factors
            exploration_bonus = c * np.sqrt(np.log(total_visits) / (visit_count + 1))
            novelty_bonus = 1.0 / (visit_count + 1)  # Encourage trying less-used actions
            
            ucb_scores[action] = q_value + exploration_bonus + novelty_bonus
        
        # Use softmax selection on UCB scores
        if np.random.random() < self._epsilon:
            # Random exploration
            selected_action = np.random.choice(available_actions)
        else:
            # Softmax selection
            probabilities = self._softmax(ucb_scores)
            actions = list(probabilities.keys())
            probs = list(probabilities.values())
            selected_action = np.random.choice(actions, p=probs)

        # Update action count
        self._action_counts[current_state][selected_action] += 1
        
        # Decay temperature and epsilon
        self._temperature = max(self._min_temperature, 
                              self._temperature * self._temperature_decay)
        self._epsilon = max(0.01, self._epsilon * 0.995)

        return selected_action  
    
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

    # def _store_experience(self, current_state: str, action: int, reward: float, new_state: str, done: bool) -> None:
    #     """
    #     Store experience tuple in buffer.
    #     """
    #     self._experience_buffer.append((current_state, action, reward, new_state, done))

    # def _learn_from_experience(self) -> None:
    #     """
    #     Learn from stored experiences.
    #     """
    #     if len(self._experience_buffer) < self._min_experience_size:
    #         return
        
    #     batch = random.sample(self._experience_buffer, 
    #                           min(self._batch_size, len(self._experience_buffer)))
        
    #     for current_state, action, reward, new_state, done in batch:
    #         # Initialize Q-values for new states
    #         if current_state not in self._q_table:
    #             self._q_table[current_state] = {i: 0.0 for i in range(len(self._rules))}
    #         if new_state not in self._q_table:
    #             self._q_table[new_state] = {i: 0.0 for i in range(len(self._rules))}
            
    #         current_q = self._q_table[current_state][action]
    #         if done:
    #             target_q = reward
    #         else:
    #             next_max_q = max(self._q_table[new_state].values())
    #             target_q = reward + self._discount_factor * next_max_q
                
    #         self._q_table[current_state][action] = (
    #             (1 - self._learning_rate) * current_q +
    #             self._learning_rate * target_q
    #         )
    
    def train(self, word: str, target_word: str) -> None:
        """
        Train the model.
        """
        normalized_word = normalize_word(word)
        normalized_target = normalize_word(target_word)
        best_episode_reward = float('-inf')
        best_rule_sequence = None

        # train for multiple episodes
        for episode in range(self._curriculum_episodes):
            # reset episode state
            self._bag_of_words = [normalized_word]
            self._target_word = normalized_target
            self._bag_of_results = []
            self._last_rules = []
            self._previous_min_distance = float('inf')
            episode_reward = 0

            # run episode until terminal state
            while True:
                # get current state and choose action
                current_state = self._get_state()
                action = self._choose_action()

                # terminal condition
                if action == -1:
                    break
                
                # apply rule
                self._apply_rule(action)
                new_state = self._get_state()
                reward = self._calculate_reward()
                episode_reward += reward

                # check if target reached
                done = self._target_word in self._bag_of_results

                # store experience
                self._store_experience(current_state, action, reward, new_state, done)

                # learn from experience
                self._learn_from_experience()

                if done:
                    # store success pattern
                    if episode_reward > best_episode_reward:
                        best_episode_reward = episode_reward
                        best_rule_sequence = self._last_rules.copy()
                    break

                # Adaptive curriculum adjustment
                self._adjust_curriculum(episode, episode_reward)
                
                # Early stopping if good solution found
                if best_episode_reward > 0.9:  # threshold for "good enough"
                    consecutive_good = 0
                    consecutive_good += 1
                    if consecutive_good >= 5:  # early stopping after 5 good episodes
                        break
                else:
                    consecutive_good = 0

        # save best rule sequence
        if best_rule_sequence:
            key = f"{normalized_word}->{normalized_target}"
            self._success_pattern[key] = best_rule_sequence

    def _adjust_curriculum(self, episode: int, episode_reward: float):
        """
        Adjust curriculum based on learning progress.
        """
        progress = episode / self._curriculum_episodes
        
        # Adjust learning parameters based on progress and performance
        if progress < 0.3:  # Early stage
            self._learning_rate = max(0.1, self._learning_rate * 1.01)
            self._epsilon = min(0.3, self._epsilon * 1.02)
        elif progress < 0.7:  # Middle stage
            self._learning_rate = self._learning_rate * 0.995
            self._epsilon = self._epsilon * 0.99
        else:  # Late stage
            self._learning_rate = max(0.01, self._learning_rate * 0.99)
            self._epsilon = max(0.01, self._epsilon * 0.98)
        
        # Adjust based on episode performance
        if episode_reward > 0:
            self._discount_factor = min(0.99, self._discount_factor * 1.001)
        else:
            self._discount_factor = max(0.8, self._discount_factor * 0.999)

    def _update_q_table(self, current_state: str, action: int, reward: float, new_state: str) -> bool:
        """
        Update the Q-table.
        """
        if current_state not in self._q_table:
            self._q_table[current_state] = {
                i: 0.0 for i in range(len(self._rules))
            }
        if new_state not in self._q_table:
            self._q_table[new_state] = {
                i: 0.0 for i in range(len(self._rules))
            }

        current_q = self._q_table[current_state][action]
        next_max_q = max(self._q_table[new_state].values()) if self._q_table[new_state] else 0

        # Q-learning update formula
        new_q = current_q + self._learning_rate * (
            reward + 
            self._discount_factor * next_max_q - current_q
        )
        self._q_table[current_state][action] = new_q

        # Terminal condition if target reached
        if self._target_word in self._bag_of_results:
            return True
        return False
