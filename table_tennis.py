import pygame
import random
import numpy as np
import pickle
import os

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 800
HEIGHT = 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Table Tennis")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle settings
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
PADDLE_SPEED = 5

# Ball settings
BALL_SIZE = 10
BALL_SPEED_X = 2
BALL_SPEED_Y = 2

# AI settings
STATE_SIZE = 5  # Ball x, y, velocity x, velocity y, paddle y
ACTION_SIZE = 3  # Up, Down, Stay
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 0.3

# Training parameters
TRAINING_EPISODES = 20000
RENDER_TRAINING = False
EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.01

# Save/Load parameters
SAVE_FILE = 'q_table.pkl'
TRAIN_MODE = True  # Set this to False to play without training

# Initialize Q-table
STATE_BINS = (10, 10, 2, 2, 10)

# Initialize clock
clock = pygame.time.Clock()

class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move(self, dy):
        self.rect.y += dy
        self.rect.clamp_ip(screen.get_rect())

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rect = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.dx = BALL_SPEED_X * random.choice((1, -1))
        self.dy = BALL_SPEED_Y * random.choice((1, -1))

    def move(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.dy *= -1

    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)

def get_state(ball, ai_paddle):
    return np.array([
        ball.rect.centerx / WIDTH,
        ball.rect.centery / HEIGHT,
        ball.dx / BALL_SPEED_X,
        ball.dy / BALL_SPEED_Y,
        ai_paddle.rect.centery / HEIGHT
    ])

def discretize_state(state, bins=STATE_BINS):
    discrete_state = []
    for i, val in enumerate(state):
        discrete_state.append(np.digitize(val, np.linspace(0, 1, bins[i])) - 1)
    return tuple(discrete_state)

def choose_action(state, epsilon):
    discrete_state = discretize_state(state)
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    else:
        return np.argmax(q_table[discrete_state])

def update_q_table(state, action, reward, next_state):
    discrete_state = discretize_state(state)
    discrete_next_state = discretize_state(next_state)
    current_q = q_table[discrete_state + (action,)]
    next_max_q = np.max(q_table[discrete_next_state])
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - current_q)
    q_table[discrete_state + (action,)] = new_q

def run_episode(render=False, train=True):
    global player_paddle, ai_paddle, ball

    if not hasattr(run_episode, "initialized"):
        player_paddle = Paddle(50, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        ai_paddle = Paddle(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        ball = Ball()
        run_episode.initialized = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return None, 0

    # AI movement
    state = get_state(ball, ai_paddle)
    action = choose_action(state, epsilon if train else 0)

    if action == 0:  # Move up
        ai_paddle.move(-PADDLE_SPEED)
    elif action == 1:  # Move down
        ai_paddle.move(PADDLE_SPEED)

    # Ball movement
    ball.move()

    # Collision detection
    if ball.rect.colliderect(player_paddle.rect) or ball.rect.colliderect(ai_paddle.rect):
        ball.dx *= -1

    # Scoring and rewards
    reward = 0
    if ball.rect.left <= 0:
        reward = -1
        ball.reset()
    elif ball.rect.right >= WIDTH:
        reward = 1
        ball.reset()
    else:
        # Small reward for keeping the paddle close to the ball
        reward = 0.1 * (1 - abs(ball.rect.centery - ai_paddle.rect.centery) / HEIGHT)

    # Update Q-table if training
    if train:
        next_state = get_state(ball, ai_paddle)
        update_q_table(state, action, reward, next_state)

    # Render
    if render:
        screen.fill(BLACK)
        player_paddle.draw()
        ai_paddle.draw()
        ball.draw()
        pygame.display.flip()
        clock.tick(60)

    return True, reward  # Continue playing and return the reward

def save_q_table(q_table):
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(q_table, f)

def load_q_table():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'rb') as f:
            return pickle.load(f)
    return np.zeros(STATE_BINS + (ACTION_SIZE,))

# Load existing Q-table if available
q_table = load_q_table()

if TRAIN_MODE:
    # Training phase
    epsilon = EPSILON
    total_rewards = []
    for episode in range(TRAINING_EPISODES):
        episode_reward = 0
        done = False
        while not done:
            result, reward = run_episode(render=RENDER_TRAINING, train=True)
            if result is None:
                done = True
            else:
                episode_reward += reward

        if episode_reward is None:
            break

        total_rewards.append(episode_reward)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            avg_reward = sum(total_rewards[-100:]) / len(total_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Save the trained Q-table
    save_q_table(q_table)
    print("Training complete. Q-table saved. Game will start automatically.")
else:
    print("Loaded existing Q-table. Game will start automatically.")

# Game loop (playing against trained AI)
running = True
while running:
    result, _ = run_episode(render=True, train=False)
    if result is None:
        running = False

    # Player movement (controlled by arrow keys)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_paddle.move(-PADDLE_SPEED)
    if keys[pygame.K_DOWN]:
        player_paddle.move(PADDLE_SPEED)

pygame.quit()