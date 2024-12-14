from typing import TypedDict, List
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from .rules import Rule
import os
from .rule_swapper import RuleSwapper

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class WordEntry(TypedDict):
    word: str
    target: str

class ReinforcementLearning:
    def __init__(self, rules: List[Rule], num_words: int):
        self._num_rules = len(rules)
        self._num_words = num_words
        self._rules = rules
        self._rule_swapper = RuleSwapper(self._num_rules)

        # state representation
        self._state_size = self._num_rules * self._num_words
        self._action_size = self._num_rules * self._num_rules

        # hyperparameters
        self._gamma = 0.95
        self._epsilon = 1.0
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._learning_rate = 0.001

        # neural network model
        self._model = self._build_model()

        # current rule order for each word
        self._current_orders = [list(range(self._num_rules)) for _ in range(self._num_words)]

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self._state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self._action_size, activation='linear')
        ])
        model.compile(loss='mse', 
                      optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate))
        return model
    
    def _get_state(self):
        # flatten the current rule order into a single state vector
        return np.array([np.concatenate(self._current_orders)])
    
    def _apply_global_action(self, action: int) -> List[List[int]]:
        """Apply a global action to modify rule ordering across all words."""
        return self._rule_swapper.apply_action(action, self._current_orders)
    
    def _evaluate_performance(self, word: str, target: str, rule_order: List[List[int]]) -> float:
        """Evaluate the performance of the rule order."""
        return 0
    
    def train(self, words: List[WordEntry], episodes: int):
        """
        Train on multiple words with their target words in a batch.

        Args:
            words: List of words and their target words.
            episodes: Number of episodes to train on.
        """  
        # Training loop
        best_performance = float('-inf')
        best_rule_order = None

        for episode in range(episodes):
            total_reward = 0

            for word, target in words:
                # get current state
                current_state = self._get_state()

                # choose action
                action = self._choose_action(current_state)
                new_rule_order = self._apply_global_action(action)
                self._current_orders = new_rule_order

                # get new state
                next_state = self._get_state()

                # calculate reward
                reward = self._evaluate_performance(word, target, new_rule_order)
                total_reward += reward

                # update model
                self._update_model(
                    current_state, 
                    action, 
                    reward, 
                    next_state, 
                    False # handle this later
                )

                # track best performance
                if reward > best_performance:
                    best_performance = reward
                    best_rule_order = new_rule_order

            # print episode summary
            print(f"Episode {episode + 1} completed. Total reward: {total_reward}")
            print(f"Best performance: {best_performance}")
            print(f"Best rule order: \n{best_rule_order}")

    def _choose_action(self, state: np.ndarray) -> int:
        """Choose an action based on the current state."""
        # epsilon-greedy policy
        if np.random.rand() < self._epsilon:
            return np.random.randint(self._action_size)
        
        # model prediction
        action_values = self._model.predict(state)
        return np.argmax(action_values)
    
    def _update_model(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Update the model using the given experience."""
        target = reward + self._gamma * np.max(self._model.predict(next_state))
        target_full = self._model.predict(state)
        target_full[0][action] = target
        self._model.fit(state, target_full, epochs=1, verbose=0)

        # decay epsilon
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

