import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim)
        )

    def forward(self, x):
        return self.net(x)

class SACAgent:
    def __init__(self, s_dim, a_dim):
        self.q = QNet(s_dim, a_dim)
        self.opt = optim.Adam(self.q.parameters(), lr=3e-4)
        self.gamma = 0.99

    def act(self, state):
        s = torch.FloatTensor(state)
        probs = torch.softmax(self.q(s), dim=-1)
        return torch.multinomial(probs, 1).item()

    def learn(self, s, a, r, ns, done):
        s = torch.FloatTensor(s)
        ns = torch.FloatTensor(ns)

        q_val = self.q(s)[a]
        with torch.no_grad():
            target = r + self.gamma * torch.max(self.q(ns)) * (1-done)

        loss = (q_val - target).pow(2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
