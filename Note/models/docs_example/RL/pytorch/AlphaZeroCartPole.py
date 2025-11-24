import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from Note.nn import RL_pytorch
from Note.RL.rl.mcts import Node_, run_mcts_search_, select_action_after_search_


# ==============================================================
#  Network definition (shared trunk for policy + value heads)
# ==============================================================
class PVNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)   # Outputs logits for π(a|s)
        self.value_head  = nn.Linear(hidden, 1)            # Outputs state value v(s)

    def forward(self, x):
        # Automatically add batch dimension if input is a single state vector
        x = torch.FloatTensor(x).unsqueeze(0) if x.ndim == 1 else torch.FloatTensor(x)
        h = self.trunk(x)
        logits = self.policy_head(h)                     # (B, action_dim)
        value  = torch.tanh(self.value_head(h))          # (B, 1) → squeezed to (B,)
        return logits, value.squeeze(-1)


# ==============================================================
#  Inherit from your original RL_pytorch framework
# ==============================================================
class AlphaZeroCartPole(RL_pytorch):
    def __init__(self, env_name="CartPole-v1"):
        super().__init__()

        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Device selection (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PVNet(self.state_dim, self.action_dim).to(self.device)

        # Tell the framework where the trainable parameters are
        self.param = self.net.parameters()

        # MCTS hyper-parameters
        self.num_simulations = 400      # Number of MCTS simulations per move
        self.c_puct = 2.0               # Exploration constant in PUCT formula
        self.mcts_temp = 1.0            # Temperature for visit-count → policy conversion

        # Store the MCTS-improved policy (visit distribution) for the current step
        self.current_pi = None
    
    def policy_network(self,s):
        logits, value = self.net(s)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
        return [(a, probs[a]) for a in range(self.action_dim)], value

    # Core override: the framework calls action() to choose a move → we use MCTS
    def action(self, s):
        """
        s: numpy array representing the current state
        Returns a torch tensor with the chosen action (int64)
        """
        # Create root node (prior probability is irrelevant for the root)
        root_node = Node_(parent=None, prior_p=0.0)

        # Run MCTS search starting from the current state
        root_node = run_mcts_search_(
            node=root_node,
            root_state=s,
            game=self.env,
            policy_network=self.net,      # The same network is used for both policy and value
            value_network=None,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct
        )

        # Convert visit counts → final action + improved policy π
        action, pi = select_action_after_search_(root_node, temperature=self.mcts_temp)

        # Save the MCTS policy distribution (used later in loss computation)
        self.current_pi = torch.tensor(pi, dtype=torch.float32, device=self.device)

        # Return action as torch tensor (required by your RL_pytorch framework)
        return torch.tensor(action, dtype=torch.int64, device=self.device)

    # Loss function called by the framework → AlphaZero-style loss
    def __call__(self, s, a, next_s, r, d):
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device).view(-1, 1)
        r = torch.FloatTensor(r).to(self.device).view(-1, 1)
    
        logits, v_pred = self.net(s)
    
        log_probs = F.log_softmax(logits, dim=-1)
        
        policy_loss = -torch.mean(torch.sum(self.current_pi * log_probs, dim=1))
    
        value_loss = F.mse_loss(v_pred, r.squeeze(-1))
    
        l2 = sum(p.pow(2.0).sum() for p in self.net.parameters()) * 1e-4
    
        return policy_loss + value_loss + l2