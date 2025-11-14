import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

pygame.init()

display = pygame.display.set_mode(
    (1280,720),
    pygame.DOUBLEBUF |
    pygame.OPENGL |
    pygame.HIDDEN 
)

print(" OPENGL context created")

