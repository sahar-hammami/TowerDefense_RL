import random

class Enemy:
    def __init__(self):
        self.position = 0.0  # Position de 0 (départ) à 1 (base)
        self.speed = 0.02    # Avance à chaque step

    def update(self):
        self.position += self.speed
        if self.position > 1.0:
            self.position = 1.0

class GameEngine:
    def __init__(self):
        self.towers = []
        self.enemies = []
        self.base_hp = 100
        self.gold = 50
        self.step_count = 0

    def reset(self):
        self.towers = []
        self.enemies = [Enemy() for _ in range(3)]
        self.base_hp = 100
        self.gold = 50
        self.step_count = 0

    def update(self):
        self.step_count += 1

        # Déplacement des ennemis
        for e in self.enemies:
            e.update()

        # Si un ennemi atteint la base
        for e in self.enemies:
            if e.position >= 1.0:
                self.base_hp -= 10
                self.enemies.remove(e)
                self.gold += 5

class Tower:
    def __init__(self):
        self.damage = 5
