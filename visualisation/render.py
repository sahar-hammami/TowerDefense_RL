import pygame
import sys
import torch
from env.td_env import TowerDefenseEnv
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

# -----------------------------
# 1️⃣ Choix de l'agent à visualiser
# -----------------------------
AGENT_TYPE = "PPO"   # "PPO" ou "SAC"

# -----------------------------
# 2️⃣ Initialisation Pygame
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 800, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f"Tower Defense RL - {AGENT_TYPE}")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# -----------------------------
# 3️⃣ Charger l'environnement
# -----------------------------
env = TowerDefenseEnv()

# -----------------------------
# 4️⃣ Charger l'agent
# -----------------------------
state_dim = 5
action_dim = 2

if AGENT_TYPE == "PPO":
    agent = PPOAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load("ppo_model.pth"))
    agent.model.eval()
elif AGENT_TYPE == "SAC":
    agent = SACAgent(state_dim, action_dim)
    agent.q.load_state_dict(torch.load("sac_model.pth"))
    agent.q.eval()
else:
    raise ValueError("AGENT_TYPE doit être 'PPO' ou 'SAC'")

# -----------------------------
# 5️⃣ Fonction de dessin
# -----------------------------
def draw():
    screen.fill((30, 30, 30))

    # Base
    pygame.draw.rect(screen, (0, 255, 0), (700, 120, 50, 60))

    # Tours
    for i, t in enumerate(env.game.towers):
        pygame.draw.rect(screen, (0, 0, 255), (200+i*40, 180, 30, 30))

    # Ennemis
    for e in env.game.enemies:
        x = int(50 + e.position*600)
        pygame.draw.circle(screen, (255, 0, 0), (x, 120), 10)

    # Infos
    text = font.render(
        f"Base HP: {env.game.base_hp} | Gold: {env.game.gold} | Enemies: {len(env.game.enemies)}",
        True, (255, 255, 255)
    )
    screen.blit(text, (10, 10))

    pygame.display.flip()

# -----------------------------
# 6️⃣ Boucle principale
# -----------------------------
state = env.reset()
done = False

while not done:
    # Quitter si on ferme la fenêtre
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Action de l'agent
    if AGENT_TYPE == "PPO":
        action, _ = agent.act(state)
    else:  # SAC
        action = agent.act(state)

    # Step dans l'environnement
    next_state, reward, done = env.step(action)
    state = next_state

    # Dessin
    draw()
    clock.tick(10)  # 10 FPS
