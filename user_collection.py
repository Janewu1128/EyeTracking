
import pygame
import time
import csv
import random

pygame.init()


screen_width, screen_height = 1440, 900
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# initialize the position and speed
point_position = [random.randint(0, screen_width), random.randint(0, screen_height)]
point_speed = [5, 3]

# record the position in csv file
with open('point_positions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'x', 'y'])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # press Esc to stop
                    running = False

        point_position[0] += point_speed[0]
        point_position[1] += point_speed[1]

        if point_position[0] <= 0 or point_position[0] >= screen_width:
            point_speed[0] = -point_speed[0]
        if point_position[1] <= 0 or point_position[1] >= screen_height:
            point_speed[1] = -point_speed[1]

        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), point_position, 10)
        pygame.display.flip()
        timestamp = time.time()
        writer.writerow([timestamp, point_position[0], point_position[1]])
        clock.tick(60)

pygame.quit()
