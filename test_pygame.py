import pygame
pygame.init()

screen = pygame.display.set_mode((400,300))
pygame.display.set_caption("TEST PYGAME")

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    screen.fill((0, 100, 200))
    pygame.display.flip()

pygame.quit()
