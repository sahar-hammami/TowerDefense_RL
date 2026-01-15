import torch
import torch.nn as nn
import torch.optim as optim

class PPO(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, a_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, s_dim, a_dim):
        self.model = PPO(s_dim, a_dim)
        self.opt = optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99

    def act(self, state):
        s = torch.FloatTensor(state)
        logits, _ = self.model(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a)

    def learn(self, traj):
        states, actions, rewards, logps = zip(*traj)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        logps = torch.stack(logps)

        logits, values = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_logps = dist.log_prob(actions)

        advantage = returns - values.squeeze()
        loss = -(new_logps * advantage.detach()).mean() + advantage.pow(2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
