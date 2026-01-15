from env.td_env import TowerDefenseEnv
from agents.ppo_agent import PPOAgent
import pygame, sys
import torch

# -----------------------------
# 1️⃣ Initialisation Pygame (optionnelle)
# -----------------------------
visualize = True  # True pour voir l'IA jouer pendant l'entraînement

if visualize:
    pygame.init()
    WIDTH, HEIGHT = 800, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tower Defense RL PPO Training")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

# -----------------------------
# 2️⃣ Environnement et Agent
# -----------------------------
env = TowerDefenseEnv()
agent = PPOAgent(5, 2)  # correct pour ton agent actuel


# -----------------------------
# 3️⃣ Fonction de dessin (Pygame)
# -----------------------------
def draw():
    screen.fill((30,30,30))
    # Base
    pygame.draw.rect(screen, (0,255,0), (700,120,50,60))
    # Tours
    for i, t in enumerate(env.game.towers):
        pygame.draw.rect(screen, (0,0,255), (200+i*40,180,30,30))
    # Ennemis
    for e in env.game.enemies:
        x = int(50 + e.position*600)
        pygame.draw.circle(screen, (255,0,0), (x,120), 10)
    # Infos
    text = font.render(
        f"Base HP: {env.game.base_hp} | Gold: {env.game.gold} | Enemies: {len(env.game.enemies)}",
        True, (255,255,255)
    )
    screen.blit(text,(10,10))
    pygame.display.flip()

# -----------------------------
# 4️⃣ Boucle d'entraînement
# -----------------------------
episodes = 10
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    traj = []

    while not done:
        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # Action de l'agent
        action, log_prob = agent.act(state)
        next_state, reward, done = env.step(action)
        traj.append((state, action, reward, log_prob))
        agent.learn(traj)  # apprentissage en ligne
        state = next_state
        total_reward += reward

        if visualize:
            draw()
            clock.tick(10)

    print(f"PPO Episode {ep+1} Reward: {total_reward:.2f}")

# -----------------------------
# 5️⃣ Sauvegarde du modèle
# -----------------------------
torch.save(agent.model.state_dict(), "ppo_model.pth")
print("✅ PPO model saved as ppo_model.pth")
