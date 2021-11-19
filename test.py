import pygame
from pygame.locals import * 


image = pygame.image.load('car.png')
image_rect = image.get_rect()
done = True
WHITE = (0, 0, 0)
clock= pygame.time.Clock()
size = [400, 300]
screen= pygame.display.set_mode(size)
while done:
    clock.tick(10)
    

    screen.fill(WHITE)
    screen.blit(image, image_rect)
    