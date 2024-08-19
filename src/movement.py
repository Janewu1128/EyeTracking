import pygame
import sys
import random

# Initialize Pygame and the drawing window
pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Colors and initial position
red = (255, 0, 0)
x, y = random.randint(0, screen_width), random.randint(0, screen_height)
# Initial velocities, determine the direction and speed of the dot's movement.
vx, vy = random.randint(-10, 10), random.randint(-10, 10) 

def move_dot():
    global x, y, vx, vy
    speed_change_chance = 0.1  # Chance to change speed each frame

    # Randomly change velocity to create varied movement
    if random.random() < speed_change_chance:
        vx, vy = random.randint(-10, 10), random.randint(-10, 10)

    # Update position based on velocity
    x += vx
    y += vy

    # Boundary checking to reverse direction if hitting a wall
    if x < 0 or x > screen_width:
        vx *= -1  # Reverse horizontal direction
        x = max(0, min(x, screen_width))  # Ensure the dot stays within the screen

    if y < 0 or y > screen_height:
        vy *= -1  # Reverse vertical direction
        y = max(0, min(y, screen_height))  # Ensure the dot stays within the screen

    # Update the screen
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, red, (x, y), 20)
    pygame.display.flip()

    # Print the coordinates
    print(f"Dot coordinates: (x={x}, y={y})")

try:
    while True:
        move_dot()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        clock.tick(30)  # Limit to 30 frames per second to make movement smoother
finally:
    pygame.quit()
