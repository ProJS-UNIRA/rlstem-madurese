from typing import List
import numpy as np

class RuleSwapper:
    """Handles rule swapping operations for the stemming agent."""
    
    def __init__(self, num_rules: int):
        self._num_rules = num_rules
    
    def apply_action(self, action: int, current_orders: np.ndarray) -> np.ndarray:
        """Apply a global action to modify rule ordering across all words."""
        self._validate_action(action)
        swap_type, i, j = self._decode_action(action)
        new_orders = np.array([order.copy() for order in current_orders])
        
        swap_functions = {
            0: self._global_swap,
            1: self._probabilistic_swap,
            2: self._sliding_window_swap,
            3: self._group_swap
        }
        
        new_orders = swap_functions[swap_type](new_orders, i, j)
        self._validate_orders(new_orders)
        return new_orders

    def _validate_action(self, action: int) -> None:
        """Validate action is within acceptable range."""
        max_actions = 4 * self._num_rules * self._num_rules  # 4 swap types
        if not 0 <= action < max_actions:
            raise ValueError(f"Action {action} out of range [0, {max_actions})")

    def _decode_action(self, action: int) -> tuple[int, int, int]:
        """Decode action into swap_type and indices."""
        swap_type = action // (self._num_rules * self._num_rules)
        i = (action % (self._num_rules * self._num_rules)) // self._num_rules
        j = action % self._num_rules
        
        if not (0 <= i < self._num_rules and 0 <= j < self._num_rules):
            raise ValueError(f"Invalid rule indices: i={i}, j={j}")
        
        return swap_type, i, j

    def _global_swap(self, orders: np.ndarray, i: int, j: int) -> np.ndarray:
        """Perform global swap of rules at positions i and j."""
        for word_orders in orders:
            word_orders[i], word_orders[j] = word_orders[j], word_orders[i]
        return orders

    def _probabilistic_swap(self, orders: np.ndarray, i: int, j: int) -> np.ndarray:
        """Perform probabilistic swap of rules at positions i and j."""
        for word_orders in orders:
            if np.random.random() < 0.5:
                word_orders[i], word_orders[j] = word_orders[j], word_orders[i]
        return orders

    def _sliding_window_swap(self, orders: np.ndarray, i: int, j: int) -> np.ndarray:
        """Move rule i to position j."""
        for word_orders in orders:
            rule = word_orders.pop(i)
            word_orders.insert(j, rule)
        return orders

    def _group_swap(self, orders: np.ndarray, i: int, j: int) -> np.ndarray:
        """Swap blocks of rules."""
        block_size = min(2, self._num_rules // 2)
        i_block = i - (i % block_size)
        j_block = j - (j % block_size)
        
        for word_orders in orders:
            for offset in range(block_size):
                if i_block + offset < len(word_orders) and j_block + offset < len(word_orders):
                    word_orders[i_block + offset], word_orders[j_block + offset] = \
                        word_orders[j_block + offset], word_orders[i_block + offset]
        return orders

    def _validate_orders(self, orders: np.ndarray) -> None:
        """Validate the integrity of rule orders."""
        for order in orders:
            if len(order) != self._num_rules:
                raise ValueError("Rule order length changed unexpectedly")
            if sorted(order) != list(range(self._num_rules)):
                raise ValueError("Rules lost or duplicated in ordering") 