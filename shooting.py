import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 800
HEIGHT = 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Shooting Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class Character:
    def __init__(self, image_path, x, y):
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


# Player
player_x = WIDTH // 2
player_y = HEIGHT - 100
player_speed = 5
player = Character('./charlet/lulu.jpg', player_x, player_y)
player_alive = True

# Bullet
bullet_width = 5
bullet_height = 10
bullets = []
bullet_speed = 7
shoot_delay = 500  # Delay between shots in milliseconds
last_shot = pygame.time.get_ticks()

# Enemy
enemies = []
enemy_speed = 2

# Game loop
running = True
clock = pygame.time.Clock()

# Font for game over text
font = pygame.font.Font(None, 74)

while running:
    clock.tick(60)  # 60 FPS

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and not player_alive:
                # Restart game
                player_alive = True
                player.rect.x = WIDTH // 2
                player.rect.y = HEIGHT - 100
                bullets = []
                enemies = []

    if player_alive:
        # Player movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player.rect.left > 0:
            player.rect.x -= player_speed
        if keys[pygame.K_RIGHT] and player.rect.right < WIDTH:
            player.rect.x += player_speed

        # Automatic shooting
        current_time = pygame.time.get_ticks()
        if current_time - last_shot > shoot_delay:
            bullet_x = player.rect.centerx - bullet_width // 2
            bullet_y = player.rect.top
            bullets.append(pygame.Rect(bullet_x, bullet_y, bullet_width, bullet_height))
            last_shot = current_time

        # Move bullets
        for bullet in bullets[:]:
            bullet.y -= bullet_speed
            if bullet.bottom < 0:
                bullets.remove(bullet)

        # Spawn enemies
        if random.randint(1, 60) == 1:
            enemy_x = random.randint(0, WIDTH - 50)  # Assuming enemy width is roughly 50
            enemy = Character('./charlet/lulu.jpg', enemy_x, 0)
            enemies.append(enemy)

        # Move enemies
        for enemy in enemies[:]:
            enemy.rect.y += enemy_speed
            if enemy.rect.top > HEIGHT:
                enemies.remove(enemy)

        # Collision detection: bullets and enemies
        for enemy in enemies[:]:
            for bullet in bullets[:]:
                if enemy.rect.colliderect(bullet):
                    offset_x = bullet.x - enemy.rect.x
                    offset_y = bullet.y - enemy.rect.y
                    if enemy.mask.overlap(pygame.mask.Mask.fill(pygame.mask.Mask((bullet_width, bullet_height)), 1),
                                          (offset_x, offset_y)):
                        enemies.remove(enemy)
                        bullets.remove(bullet)
                        break

        # Collision detection: player and enemies
        for enemy in enemies[:]:
            if player.rect.colliderect(enemy.rect):
                offset_x = enemy.rect.x - player.rect.x
                offset_y = enemy.rect.y - player.rect.y
                if player.mask.overlap(enemy.mask, (offset_x, offset_y)):
                    player_alive = False
                    break

    # Draw everything
    window.fill(BLACK)

    if player_alive:
        player.draw(window)
        for bullet in bullets:
            pygame.draw.rect(window, WHITE, bullet)
        for enemy in enemies:
            enemy.draw(window)
    else:
        game_over_text = font.render("Game Over", True, WHITE)
        restart_text = font.render("Press R to Restart", True, WHITE)
        window.blit(game_over_text,
                    (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - game_over_text.get_height() // 2))
        window.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + restart_text.get_height()))

    pygame.display.update()

# Quit the game
pygame.quit()