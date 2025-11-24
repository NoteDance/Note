import tensorflow as tf
import numpy as np
import math


class BaseNode:
    """
    The base class for an MCTS Node.
    Uses __slots__ for memory efficiency.
    """
    # __slots__ tells Python to not use a __dict__, saving memory.
    __slots__ = ['parent', 'children', 'visit_count', 'value_sum', 'prior_p', '_state']
    
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.visit_count = tf.cast(0, tf.int64)
        self.value_sum = tf.cast(0.0, tf.float32)
        self.prior_p = prior_p
        self._state = None # Lazily set state

    def get_value(self):
        """ (Q) Calculate the average value of this node """
        if self.visit_count == 0:
            return 0
        return self.value_sum / tf.cast(self.visit_count, self.value_sum.dtype)

    def is_expanded(self):
        """ Check if this node has been fully expanded """
        return len(self.children) > 0

    def select_child(self, c_puct):
        """ **MCTS Step 1: Select** (UCT algorithm) """
        best_score = -float('inf')
        best_action = None
        best_child = None
        sqrt_total_visits = tf.sqrt(self.visit_count)

        for action, child_node in self.children.items():
            q_value = child_node.get_value()
            n_visits = child_node.visit_count
            p_value = child_node.prior_p
            u_score = c_puct * p_value * (sqrt_total_visits / (1 + n_visits))
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_node
        
        return best_action, best_child

    def expand(self, state, actions_and_priors):
        """ **MCTS Step 2: Expand** """
        self._state = state
        for action, prior_p in actions_and_priors:
            if action not in self.children:
                # IMPORTANT: Create instances of its *own* class
                # We use type(self) to ensure subclasses create the correct Node type
                self.children[action] = type(self)(parent=self, prior_p=prior_p)

    def backpropagate(self, value):
        """ **MCTS Step 4: Backpropagate** (Standard version) """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent    # Parent node
        self.children = {}      # Child nodes (a map from action -> Node)
        
        self.visit_count = tf.cast(0, tf.int64)    # (N) Visit count
        self.value_sum = tf.cast(0.0, tf.float32)    # (W) Total value (or Q-value in MuZero)
        self.prior_p = prior_p    # (P) Prior probability (from the policy network)
        
        self._state = None # The (hidden) state corresponding to this node, set only when needed

    def get_value(self):
        """ (Q) Calculate the average value of this node """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """ Check if this node has been fully expanded """
        return len(self.children) > 0

    def select_child(self, c_puct):
        """
        **MCTS Step 1: Select**
        Use the UCT (Upper Confidence Bound for Trees) algorithm to select a child.
        This is AlphaZero's UCT formula: Q(s,a) + U(s,a)
        U(s,a) = c_puct * P(s,a) * (sqrt(N(s)) / (1 + N(s,a)))
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        # self.visit_count is the parent's N(s)
        sqrt_total_visits = tf.sqrt(self.visit_count)

        for action, child_node in self.children.items():
            # Q(s,a) = child_node.get_value()
            q_value = child_node.get_value()
            
            # N(s,a) = child_node.visit_count
            n_visits = child_node.visit_count
            
            # P(s,a) = child_node.prior_p
            p_value = child_node.prior_p

            # U(s,a)
            u_score = c_puct * p_value * (sqrt_total_visits / (1 + n_visits))
            
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_node
        
        return best_action, best_child

    def expand(self, state, actions_and_priors):
        """
        **MCTS Step 2: Expand**
        Create one or more child nodes from this node.
        'actions_and_priors' is a list of (action, prior_p) from the policy network.
        """
        self._state = state # Store this state, so the value network can evaluate it
        
        for action, prior_p in actions_and_priors:
            if action not in self.children:
                self.children[action] = Node(parent=self, prior_p=prior_p)

    def backpropagate(self, value):
        """
        **MCTS Step 4: Backpropagate**
        Propagate the evaluated 'value' back up the path to the root.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            # Note: For 2-player games, the value should be inverted (value = -value)
            # We assume here for simplicity that the value is already from the correct perspective.


def run_mcts_search(node, root_state, game, policy_network, value_network, num_simulations, c_puct):
    """
    Runs an MCTS search and returns the optimized policy after "thinking".
    
    :param root_state: The current state of the game/environment (or MuZero's hidden state s0)
    :param game: A game object with methods like .get_legal_actions() and .is_terminal()
    :param policy_network: f_policy(state) -> [(action, prior_p), ...]
    :param value_network: f_value(state) -> value
    :param num_simulations: The number of "thinking" iterations to run (e.g., 800)
    :param c_puct: The exploration constant in the UCT algorithm
    :return: The root node (can be used to get visit counts)
    """

    # Create the root node
    root_node = node
        
    # Initial evaluation: get policy and value for the root node
    if value_network is not None:
        action_priors = policy_network(root_state) # Returns a list of (action, prior_p)
        value = value_network(root_state)                         # MCTS Step 3
    else:
        action_priors, value = policy_network(root_state) # Returns a list of (action, prior_p)
    
    root_node.expand(root_state, action_priors)
    root_node.backpropagate(value)                          # MCTS Step 4 (for the root node)

    # --- Start the MCTS simulation loop ---
    for _ in range(num_simulations):
        
        node = root_node
        state = root_state
        search_path = [node] # Record the search path for backpropagation

        # === Step 1: Select ===
        # As long as the node is expanded, keep going down
        while node.is_expanded():
            action, node = node.select_child(c_puct)
            state, _, done, _ = game.step(action) # Simulate the environment
            search_path.append(node)
            
            if done:
                break # Reached a terminal state

        # 'node' is now a leaf node (not yet expanded)
        leaf_node = node
        leaf_state = state

        # === Step 2 & 3: Expand & Evaluate ===
        if done:
            # If it's a terminal state, the value is determined
            if hasattr(game, 'get_game_result'):
                value = game.get_game_result(leaf_state)
            else:
                value = 0.0
        else:
            # 1. Evaluate (MCTS Step 3)
            #    Use the neural network to evaluate this leaf node
            if value_network is not None:
                action_priors = policy_network(leaf_state)
                value = value_network(leaf_state)
            else:
                action_priors, value = policy_network(leaf_state)
            
            # 2. Expand (MCTS Step 2)
            #    Add the new nodes to the tree
            leaf_node.expand(leaf_state, action_priors)

        # === Step 4: Backpropagate ===
        # Propagate the evaluated 'value' back up the search path
        leaf_node.backpropagate(value)
        
    return root_node


def select_action_after_search(root_node, temperature=1.0):
    """
    After the MCTS search is complete, select an action based on visit counts.
    """
    visit_counts = [(action, child.visit_count) 
                    for action, child in root_node.children.items()]
    
    if not visit_counts:
        return None # No feasible actions

    actions, counts = zip(*visit_counts)
    
    if temperature == 0:
        # Deterministic selection: choose the most visited
        best_action_idx = tf.argmax(counts)
        return actions[best_action_idx]
    else:
        # Exploratory selection: sample from the distribution of visit counts
        # (counts ^ (1/temperature)) / sum(counts ^ (1/temperature))
        counts_temp = tf.convert_to_tensor(counts)**(1.0 / temperature)
        counts_prob = counts_temp / tf.reduce_sum(counts_temp)
        chosen_action = np.random.choice(actions, p=counts_prob.numpy())
        return chosen_action


class BaseNode_:
    """
    The base class for an MCTS Node.
    Uses __slots__ for memory efficiency.
    """
    # __slots__ tells Python to not use a __dict__, saving memory.
    __slots__ = ['parent', 'children', 'visit_count', 'value_sum', 'prior_p', '_state']
    
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_p = prior_p
        self._state = None # Lazily set state

    def get_value(self):
        """ (Q) Calculate the average value of this node """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """ Check if this node has been fully expanded """
        return len(self.children) > 0

    def select_child(self, c_puct):
        """ **MCTS Step 1: Select** (UCT algorithm) """
        best_score = -float('inf')
        best_action = None
        best_child = None
        sqrt_total_visits = math.sqrt(self.visit_count)

        for action, child_node in self.children.items():
            q_value = child_node.get_value()
            n_visits = child_node.visit_count
            p_value = child_node.prior_p
            u_score = c_puct * p_value * (sqrt_total_visits / (1 + n_visits))
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_node
        
        return best_action, best_child

    def expand(self, state, actions_and_priors):
        """ **MCTS Step 2: Expand** """
        self._state = state
        for action, prior_p in actions_and_priors:
            if action not in self.children:
                # IMPORTANT: Create instances of its *own* class
                # We use type(self) to ensure subclasses create the correct Node type
                self.children[action] = type(self)(parent=self, prior_p=prior_p)

    def backpropagate(self, value):
        """ **MCTS Step 4: Backpropagate** (Standard version) """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class Node_:
    def __init__(self, parent, prior_p):
        self.parent = parent    # Parent node
        self.children = {}      # Child nodes (a map from action -> Node)
        
        self.visit_count = 0    # (N) Visit count
        self.value_sum = 0.0    # (W) Total value (or Q-value in MuZero)
        self.prior_p = prior_p    # (P) Prior probability (from the policy network)
        
        self._state = None # The (hidden) state corresponding to this node, set only when needed

    def get_value(self):
        """ (Q) Calculate the average value of this node """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """ Check if this node has been fully expanded """
        return len(self.children) > 0

    def select_child(self, c_puct):
        """
        **MCTS Step 1: Select**
        Use the UCT (Upper Confidence Bound for Trees) algorithm to select a child.
        This is AlphaZero's UCT formula: Q(s,a) + U(s,a)
        U(s,a) = c_puct * P(s,a) * (sqrt(N(s)) / (1 + N(s,a)))
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        # self.visit_count is the parent's N(s)
        sqrt_total_visits = math.sqrt(self.visit_count)

        for action, child_node in self.children.items():
            # Q(s,a) = child_node.get_value()
            q_value = child_node.get_value()
            
            # N(s,a) = child_node.visit_count
            n_visits = child_node.visit_count
            
            # P(s,a) = child_node.prior_p
            p_value = child_node.prior_p

            # U(s,a)
            u_score = c_puct * p_value * (sqrt_total_visits / (1 + n_visits))
            
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_node
        
        return best_action, best_child

    def expand(self, state, actions_and_priors):
        """
        **MCTS Step 2: Expand**
        Create one or more child nodes from this node.
        'actions_and_priors' is a list of (action, prior_p) from the policy network.
        """
        self._state = state # Store this state, so the value network can evaluate it
        
        for action, prior_p in actions_and_priors:
            if action not in self.children:
                self.children[action] = Node_(parent=self, prior_p=prior_p)

    def backpropagate(self, value):
        """
        **MCTS Step 4: Backpropagate**
        Propagate the evaluated 'value' back up the path to the root.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            # Note: For 2-player games, the value should be inverted (value = -value)
            # We assume here for simplicity that the value is already from the correct perspective.


def run_mcts_search_(node, root_state, game, policy_network, value_network, num_simulations, c_puct):
    """
    Runs an MCTS search and returns the optimized policy after "thinking".
    
    :param root_state: The current state of the game/environment (or MuZero's hidden state s0)
    :param game: A game object with methods like .get_legal_actions() and .is_terminal()
    :param policy_network: f_policy(state) -> [(action, prior_p), ...]
    :param value_network: f_value(state) -> value
    :param num_simulations: The number of "thinking" iterations to run (e.g., 800)
    :param c_puct: The exploration constant in the UCT algorithm
    :return: The root node (can be used to get visit counts)
    """

    # Create the root node
    root_node = node
        
    # Initial evaluation: get policy and value for the root node
    if value_network is not None:
        action_priors = policy_network(root_state) # Returns a list of (action, prior_p)
        value = value_network(root_state)                         # MCTS Step 3
    else:
        action_priors, value = policy_network(root_state) # Returns a list of (action, prior_p)                      # MCTS Step 3
    
    root_node.expand(root_state, action_priors)
    root_node.backpropagate(value)                          # MCTS Step 4 (for the root node)

    # --- Start the MCTS simulation loop ---
    for _ in range(num_simulations):
        
        node = root_node
        state = root_state
        search_path = [node] # Record the search path for backpropagation

        # === Step 1: Select ===
        # As long as the node is expanded, keep going down
        while node.is_expanded():
            action, node = node.select_child(c_puct)
            state, _, done, _ = game.step(action) # Simulate the environment
            search_path.append(node)
            
            if done:
                break # Reached a terminal state

        # 'node' is now a leaf node (not yet expanded)
        leaf_node = node
        leaf_state = state

        # === Step 2 & 3: Expand & Evaluate ===
        if done:
            # If it's a terminal state, the value is determined
            if hasattr(game, 'get_game_result'):
                value = game.get_game_result(leaf_state)
            else:
                value = 0.0
        else:
            # 1. Evaluate (MCTS Step 3)
            #    Use the neural network to evaluate this leaf node
            if value_network is not None:
                action_priors = policy_network(leaf_state)
                value = value_network(leaf_state)
            else:
                action_priors, value = policy_network(leaf_state)
            
            # 2. Expand (MCTS Step 2)
            #    Add the new nodes to the tree
            leaf_node.expand(leaf_state, action_priors)

        # === Step 4: Backpropagate ===
        # Propagate the evaluated 'value' back up the search path
        leaf_node.backpropagate(value)
        
    return root_node


def select_action_after_search_(root_node, temperature=1.0):
    """
    After the MCTS search is complete, select an action based on visit counts.
    """
    visit_counts = [(action, child.visit_count) 
                    for action, child in root_node.children.items()]
    
    if not visit_counts:
        return None # No feasible actions

    actions, counts = zip(*visit_counts)
    
    if temperature == 0:
        # Deterministic selection: choose the most visited
        best_action_idx = np.argmax(counts)
        return actions[best_action_idx]
    else:
        # Exploratory selection: sample from the distribution of visit counts
        # (counts ^ (1/temperature)) / sum(counts ^ (1/temperature))
        counts_temp = np.array(counts)**(1.0 / temperature)
        counts_prob = counts_temp / np.sum(counts_temp)
        chosen_action = np.random.choice(actions, p=counts_prob)
        return chosen_action
