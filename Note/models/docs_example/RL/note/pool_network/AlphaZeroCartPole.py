import gym
import tensorflow as tf
from Note import nn
from Note.RL.rl.mcts import Node, run_mcts_search, select_action_after_search


# ==============================================================
#  Network definition (shared trunk for policy + value heads)
# ==============================================================
class PVNet(nn.Model):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.dense1 = nn.dense(hidden, state_dim, activation='relu')
        self.dense2 = nn.dense(hidden, hidden, activation='relu')
        self.policy_head = nn.dense(action_dim, hidden)
        self.value_head = nn.dense(1, hidden, activation='tanh')

    def __call__(self,x):
        x = self.dense2(self.dense1(x))
        logits = self.policy_head(x)
        value  = self.value_head(x)
        return logits, value.squeeze(-1)


# ==============================================================
#  Inherit from your original RL_pytorch framework
# ==============================================================
class AlphaZeroCartPole(nn.RL):
    def __init__(self, processes, env_name="CartPole-v1"):
        super().__init__()

        self.env = [gym.make(env_name) for _ in range(processes)]
        self.sim_env = [gym.make(env_name) for _ in range(processes)]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.net = PVNet(self.state_dim, self.action_dim)

        # Tell the framework where the trainable parameters are
        self.param = self.net.param

        # MCTS hyper-parameters
        self.num_simulations = 400      # Number of MCTS simulations per move
        self.c_puct = 2.0               # Exploration constant in PUCT formula
        self.mcts_temp = 1.0            # Temperature for visit-count → policy conversion

        # Store the MCTS-improved policy (visit distribution) for the current step
        self.current_pi = None
    
    @tf.function(jit_compile=True)
    def policy_network(self,s):
        logits, value = self.net(s, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy().ravel()
        return [(a, probs[a]) for a in range(self.action_dim)], value

    # Core override: the framework calls action() to choose a move → we use MCTS
    def action(self, s, p):
        """
        s: numpy array representing the current state
        Returns a torch tensor with the chosen action (int64)
        """
        # Create root node (prior probability is irrelevant for the root)
        root_node = Node(parent=None, prior_p=0.0)

        # Run MCTS search starting from the current state
        root_node = run_mcts_search(
            node=root_node,
            root_state=s,
            env=self.sim_env[p],
            policy_network=self.net,      # The same network is used for both policy and value
            value_network=None,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct
        )

        # Convert visit counts → final action + improved policy π
        action, pi = select_action_after_search(root_node, temperature=self.mcts_temp)

        # Save the MCTS policy distribution (used later in loss computation)
        self.current_pi = tf.convert_to_tensor(pi, dtype=tf.float32)

        # Return action as torch tensor (required by your RL_pytorch framework)
        return action

    # Loss function called by the framework → AlphaZero-style loss
    def __call__(self, s, a, next_s, r, d):
        a=tf.expand_dims(a,axis=1)
    
        logits, v_pred = self.net(s)
    
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        policy_loss = -tf.reduce_mean(tf.reduce_sum(self.current_pi * log_probs, axis=1))
    
        value_loss = tf.keras.losses.MeanSquaredError()(r, v_pred)
    
        l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.param]) * 2 * 1e-4 
    
        return policy_loss + value_loss + l2
