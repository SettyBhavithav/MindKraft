import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
import random

# --- Constants & Colors ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240
CAMERA_POS = (SCREEN_WIDTH - CAMERA_WIDTH - 10, 10)
MENU_ITEMS = ["Shooter", "Pong", "Breakout", "Maze", "Fruit Catcher", "IQ Analyzer"]
BLACK = (30, 32, 34); WHITE = (255,255,255); RED = (255,0,0); GREEN = (0,255,0)
BLUE = (0,128,255); YELLOW = (255,255,0); PURPLE = (128,0,128); ORANGE = (255,140,0); GREY = (200,200,200)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gesture Multi-Game Platform")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 48)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

smoothed_ix = smoothed_iy = None
SMOOTH_ALPHA = 0.7

def draw_text(text, size, color, x, y, center=False):
    font_ = pygame.font.Font(None, size)
    text_surface = font_.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center: text_rect.center = (x, y)
    else: text_rect.topleft = (x, y)
    screen.blit(text_surface, text_rect)

def draw_header(game_name, gesture_hint):
    pygame.draw.rect(screen, BLUE, (0,0,SCREEN_WIDTH,60))
    draw_text(f"Game: {game_name}", 36, WHITE, 30, 15)
    draw_text(gesture_hint, 32, YELLOW, SCREEN_WIDTH//2, 30, center=True)

def draw_exit_symbol(center, hold_time, total_time):
    pygame.draw.circle(screen, RED, center, 36, 4)
    if hold_time > 0:
        angle = int(360 * min(hold_time/total_time, 1))
        pygame.draw.arc(screen, RED, (center[0]-40, center[1]-40, 80, 80), -np.pi/2, -np.pi/2 + 2*np.pi*angle/360, 9)
    draw_text("EXIT", 32, RED, center[0], center[1], center=True)

def draw_camera_feed(frame):
    camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_surf = pygame.surfarray.make_surface(cv2.resize(camera_rgb, (CAMERA_WIDTH, CAMERA_HEIGHT)).swapaxes(0, 1))
    screen.blit(camera_surf, CAMERA_POS)

def reset_smoothing():
    global smoothed_ix, smoothed_iy
    smoothed_ix = smoothed_iy = None

def get_hand_landmarks(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return hand_landmarks
    return None

def get_index_tip_xy(hand_landmarks):
    global smoothed_ix, smoothed_iy
    ix = int(hand_landmarks.landmark[8].x * SCREEN_WIDTH)
    iy = int(hand_landmarks.landmark[8].y * SCREEN_HEIGHT)
    if smoothed_ix is None: smoothed_ix, smoothed_iy = ix, iy
    else:
        smoothed_ix = int(SMOOTH_ALPHA * smoothed_ix + (1 - SMOOTH_ALPHA) * ix)
        smoothed_iy = int(SMOOTH_ALPHA * smoothed_iy + (1 - SMOOTH_ALPHA) * iy)
    return smoothed_ix, smoothed_iy

def is_pinch(hand_landmarks, threshold=0.06, history=[0]*8):
    ix, iy = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
    tx, ty = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
    d = np.hypot(ix - tx, iy - ty)
    detected = d < threshold
    history.append(int(detected)); history.pop(0)
    return sum(history) >= 4

def is_open_palm(hand_landmarks, history=[0]*10):
    base = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    tips = [np.array([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]) for i in [4,8,12,16,20]]
    detected = all(np.linalg.norm(tip - base) > 0.21 for tip in tips)
    history.append(int(detected)); history.pop(0)
    return sum(history) >= 7

def is_fist(hand_landmarks, history=[0]*10):
    base = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    tips = [np.array([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]) for i in [4,8,12,16,20]]
    detected = all(np.linalg.norm(tip - base) < 0.09 for tip in tips)
    history.append(int(detected)); history.pop(0)
    return sum(history) >= 7

# --- Introduction Screen ---
def introduction_screen():
    menu_start_time = time.time()
    exit_time = None
    while True:
        screen.fill(BLACK)
        draw_text("Welcome to the Gesture IQ Game Platform!", 54, BLUE, SCREEN_WIDTH//2, 120, center=True)
        draw_text("Control everything with your hand gestures.", 36, WHITE, SCREEN_WIDTH//2, 200, center=True)
        draw_text("Pinch: Select/Action | Open Palm (2s): Back/Hint | Fist (2s): Exit", 36, YELLOW, SCREEN_WIDTH//2, 260, center=True)
        draw_text("Pinch to continue", 40, GREEN, SCREEN_WIDTH//2, 400, center=True)
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_landmarks = get_hand_landmarks(results)
        pinch = fist = False
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pinch = is_pinch(hand_landmarks)
            fist = is_fist(hand_landmarks)
        if pinch:
            return
        if time.time() - menu_start_time > 3:
            if fist:
                if exit_time is None: exit_time = time.time()
                hold = time.time() - exit_time
                draw_exit_symbol((SCREEN_WIDTH//2, SCREEN_HEIGHT-100), hold, 2)
                if hold > 2:
                    pygame.quit(); sys.exit()
            else: exit_time = None
        draw_camera_feed(frame)
        pygame.display.flip(); clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

# --- Mode Selection ---
def mode_selection():
    pinch_cooldown = 0
    selected = 0
    options = ["Single Player", "AI Bot Opponent"]
    exit_time = None
    menu_start_time = time.time()
    while True:
        screen.fill(BLACK)
        draw_text("Select Mode", 60, BLUE, SCREEN_WIDTH//2, 120, center=True)
        draw_text("Pinch to select. Fist (2s) to exit.", 36, WHITE, SCREEN_WIDTH // 2, 180, center=True)
        for i, item in enumerate(options):
            rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 260 + i * 100, 400, 80)
            color = YELLOW if i == selected else WHITE
            pygame.draw.rect(screen, color, rect, 0 if i==selected else 3, border_radius=18)
            draw_text(item, 56, BLACK if i==selected else color, rect.centerx, rect.centery, center=True)
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_landmarks = get_hand_landmarks(results)
        ix, iy = None, None
        pinch = fist = False
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ix, iy = get_index_tip_xy(hand_landmarks)
            pinch = is_pinch(hand_landmarks)
            fist = is_fist(hand_landmarks)
        menu_rects = []
        for i in range(len(options)):
            rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 260 + i * 100, 400, 80)
            menu_rects.append(rect)
        if ix is not None and iy is not None:
            for i, rect in enumerate(menu_rects):
                if rect.collidepoint(ix, iy): selected = i
            pygame.draw.circle(screen, RED, (ix, iy), 15)
        if pinch and pinch_cooldown == 0:
            if ix is not None and iy is not None and menu_rects[selected].collidepoint(ix, iy):
                return selected  # 0: Single Player, 1: AI Bot
            pinch_cooldown = 20
        if pinch_cooldown > 0: pinch_cooldown -= 1
        if time.time() - menu_start_time > 3:
            if fist:
                if exit_time is None: exit_time = time.time()
                hold = time.time() - exit_time
                draw_exit_symbol((SCREEN_WIDTH//2, SCREEN_HEIGHT-100), hold, 2)
                if hold > 2:
                    pygame.quit(); sys.exit()
            else: exit_time = None
        draw_camera_feed(frame)
        pygame.display.flip(); clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

# --- Level Selection ---
def level_selection(game_name):
    pinch_cooldown = 0
    selected = 0
    levels = [f"Level {i+1}" for i in range(25)]
    exit_time = None
    menu_start_time = time.time()
    while True:
        screen.fill(BLACK)
        draw_text(f"Select Level for {game_name}", 54, BLUE, SCREEN_WIDTH//2, 100, center=True)
        draw_text("Pinch to select. Fist (2s) to exit.", 36, WHITE, SCREEN_WIDTH // 2, 160, center=True)
        for i in range(5):
            for j in range(5):
                idx = i*5 + j
                rect = pygame.Rect(SCREEN_WIDTH//2 - 300 + j*120, 220 + i*90, 100, 60)
                color = YELLOW if idx == selected else WHITE
                pygame.draw.rect(screen, color, rect, 0 if idx==selected else 3, border_radius=10)
                draw_text(str(idx+1), 40, BLACK if idx==selected else color, rect.centerx, rect.centery, center=True)
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_landmarks = get_hand_landmarks(results)
        ix, iy = None, None
        pinch = fist = False
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ix, iy = get_index_tip_xy(hand_landmarks)
            pinch = is_pinch(hand_landmarks)
            fist = is_fist(hand_landmarks)
        level_rects = []
        for i in range(5):
            for j in range(5):
                idx = i*5 + j
                rect = pygame.Rect(SCREEN_WIDTH//2 - 300 + j*120, 220 + i*90, 100, 60)
                level_rects.append(rect)
        if ix is not None and iy is not None:
            for idx, rect in enumerate(level_rects):
                if rect.collidepoint(ix, iy): selected = idx
            pygame.draw.circle(screen, RED, (ix, iy), 15)
        if pinch and pinch_cooldown == 0:
            if ix is not None and iy is not None and level_rects[selected].collidepoint(ix, iy):
                return selected+1
            pinch_cooldown = 20
        if pinch_cooldown > 0: pinch_cooldown -= 1
        if time.time() - menu_start_time > 3:
            if fist:
                if exit_time is None: exit_time = time.time()
                hold = time.time() - exit_time
                draw_exit_symbol((SCREEN_WIDTH//2, SCREEN_HEIGHT-100), hold, 2)
                if hold > 2:
                    pygame.quit(); sys.exit()
            else: exit_time = None
        draw_camera_feed(frame)
        pygame.display.flip(); clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

# --- IQ Analyzer (Demo: Pattern Memory Game) ---
def iq_analyzer():
    # For brevity, a single IQ game is shown. Expand with more mini-games as needed.
    pattern = [random.randint(0,3) for _ in range(3)]
    user_pattern = []
    score = 0
    exit_time = None
    menu_start_time = time.time()
    instructions = [
        "Pattern Memory IQ Test",
        "Watch the color sequence, then repeat it by pinching on the correct color.",
        "Sequence gets longer each round. Fist (2s) to exit."
    ]
    color_blocks = [(SCREEN_WIDTH//2-180, 300, RED), (SCREEN_WIDTH//2-60, 300, GREEN), (SCREEN_WIDTH//2+60, 300, BLUE), (SCREEN_WIDTH//2+180, 300, YELLOW)]
    round_num = 1
    while True:
        screen.fill(BLACK)
        draw_header("IQ Analyzer", "Pinch: Select | Fist (2s): Exit")
        for i, line in enumerate(instructions):
            draw_text(line, 36, WHITE, SCREEN_WIDTH//2, 120+i*40, center=True)
        # Show pattern
        for idx, (x, y, color) in enumerate(color_blocks):
            pygame.draw.rect(screen, color, (x-40, y-40, 80, 80))
        pygame.display.flip(); pygame.time.wait(500)
        for idx in pattern:
            x, y, color = color_blocks[idx]
            pygame.draw.rect(screen, WHITE, (x-40, y-40, 80, 80), 5)
            pygame.display.flip(); pygame.time.wait(400)
            pygame.draw.rect(screen, color, (x-40, y-40, 80, 80))
            pygame.display.flip(); pygame.time.wait(200)
        # Wait for user input
        user_pattern = []
        input_idx = 0
        while input_idx < len(pattern):
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            hand_landmarks = get_hand_landmarks(results)
            ix, iy = None, None; pinch = fist = False
            if hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                ix, iy = get_index_tip_xy(hand_landmarks)
                pinch = is_pinch(hand_landmarks)
                fist = is_fist(hand_landmarks)
            for idx2, (x, y, color) in enumerate(color_blocks):
                pygame.draw.rect(screen, color, (x-40, y-40, 80, 80))
                if ix is not None and iy is not None and pygame.Rect(x-40, y-40, 80, 80).collidepoint(ix, iy):
                    pygame.draw.rect(screen, WHITE, (x-40, y-40, 80, 80), 5)
                    if pinch:
                        user_pattern.append(idx2)
                        input_idx += 1
                        pygame.display.flip(); pygame.time.wait(300)
            if fist:
                if exit_time is None: exit_time = time.time()
                hold = time.time() - exit_time
                draw_exit_symbol((SCREEN_WIDTH//2, SCREEN_HEIGHT-100), hold, 2)
                if hold > 2:
                    return
            else: exit_time = None
            draw_camera_feed(frame)
            pygame.display.flip(); clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
        # Check pattern
        if user_pattern == pattern:
            score += 10*round_num
            draw_text("Correct!", 60, GREEN, SCREEN_WIDTH//2, SCREEN_HEIGHT//2, center=True)
            pygame.display.flip(); pygame.time.wait(1000)
            round_num += 1
            pattern.append(random.randint(0,3))
        else:
            iq_score = 80 + score // 2
            draw_text(f"Your IQ Score: {iq_score}", 60, YELLOW, SCREEN_WIDTH//2, SCREEN_HEIGHT//2, center=True)
            pygame.display.flip(); pygame.time.wait(2500)
            return

# --- Maze Game with AI Bot and Hint ---
def bfs_shortest_path(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    queue = [(start, [start])]
    visited = set([start])
    while queue:
        (r, c), path = queue.pop(0)
        if (r, c) == goal:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and maze[nr][nc]==0 and (nr, nc) not in visited:
                queue.append(((nr, nc), path+[(nr, nc)]))
                visited.add((nr, nc))
    return []

def maze_game(level=1, ai_mode=False):
    reset_smoothing()
    MAZE_ROWS, MAZE_COLS = 8 + level, 10 + level
    CELL_SIZE = min(60, SCREEN_HEIGHT // (MAZE_ROWS+2))
    maze_top_left = (100, 100)
    player_pos = [0, 0]
    ai_pos = [0, 0]
    goal_pos = [MAZE_ROWS-1, MAZE_COLS-1]
    show_hint = False
    ai_active = ai_mode
    ai_path = []
    return_to_menu_time = None
    exit_time = None
    game_start_time = time.time()
    hint_time = None
    def generate_maze(rows, cols):
        maze = [[1 for _ in range(cols)] for _ in range(rows)]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        def neighbors(r, c):
            nbs = []
            for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]: nbs.append((nr,nc))
            return nbs
        def carve(r, c):
            visited[r][c] = True; maze[r][c] = 0
            nbs = neighbors(r, c); random.shuffle(nbs)
            for nr, nc in nbs:
                if not visited[nr][nc]:
                    maze[(r+nr)//2][(c+nc)//2] = 0; carve(nr, nc)
        carve(0,0)
        maze[0][0] = 0; maze[rows-1][cols-1] = 0
        return maze
    maze = generate_maze(MAZE_ROWS, MAZE_COLS)
    running = True; move_cooldown = 0; ai_move_cooldown = 0
    start_time = time.time()
    used_hint = False
    while running:
        screen.fill(BLACK)
        draw_header(f"Maze - Level {level}", "Pinch: Move | Open palm (2s): Hint | Fist (2s): Exit")
        # Draw maze
        for r in range(MAZE_ROWS):
            for c in range(MAZE_COLS):
                rect = pygame.Rect(maze_top_left[0]+c*CELL_SIZE, maze_top_left[1]+r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze[r][c] == 1: pygame.draw.rect(screen, WHITE, rect)
        # Draw player, goal, AI
        player_rect = pygame.Rect(maze_top_left[0]+player_pos[1]*CELL_SIZE+10, maze_top_left[1]+player_pos[0]*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20)
        goal_rect = pygame.Rect(maze_top_left[0]+goal_pos[1]*CELL_SIZE+15, maze_top_left[1]+goal_pos[0]*CELL_SIZE+15, CELL_SIZE-30, CELL_SIZE-30)
        pygame.draw.rect(screen, PURPLE, goal_rect)
        pygame.draw.ellipse(screen, GREEN, player_rect)
        if ai_active:
            ai_rect = pygame.Rect(maze_top_left[0]+ai_pos[1]*CELL_SIZE+10, maze_top_left[1]+ai_pos[0]*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20)
            pygame.draw.ellipse(screen, BLUE, ai_rect)
        # Draw hint path
        if show_hint:
            path = bfs_shortest_path(maze, tuple(player_pos), tuple(goal_pos))
            for (r, c) in path:
                rect = pygame.Rect(maze_top_left[0]+c*CELL_SIZE+20, maze_top_left[1]+r*CELL_SIZE+20, CELL_SIZE-40, CELL_SIZE-40)
                pygame.draw.rect(screen, YELLOW, rect)
        # AI bot movement
        if ai_active:
            if not ai_path or ai_pos != ai_path[-1]:
                ai_path = bfs_shortest_path(maze, tuple(ai_pos), tuple(goal_pos))
            if ai_path and ai_move_cooldown == 0 and ai_pos != list(goal_pos):
                ai_pos = list(ai_path[1])
                ai_move_cooldown = max(2, 12 - level//2)
            if ai_move_cooldown > 0: ai_move_cooldown -= 1
        # Gesture and camera
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_landmarks = get_hand_landmarks(results)
        ix, iy = None, None; pinch = open_palm = fist = False
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ix, iy = get_index_tip_xy(hand_landmarks)
            pinch = is_pinch(hand_landmarks)
            open_palm = is_open_palm(hand_landmarks)
            fist = is_fist(hand_landmarks)
        # Player move
        if pinch and move_cooldown == 0 and ix is not None and iy is not None:
            rel_x = ix - maze_top_left[0]; rel_y = iy - maze_top_left[1]
            c = int(rel_x / CELL_SIZE); r = int(rel_y / CELL_SIZE)
            if 0 <= r < MAZE_ROWS and 0 <= c < MAZE_COLS and maze[r][c] == 0:
                if abs(r - player_pos[0]) + abs(c - player_pos[1]) == 1:
                    player_pos[0], player_pos[1] = r, c; move_cooldown = 10
        if move_cooldown > 0: move_cooldown -= 1
        # Win
        if player_pos == goal_pos or (ai_active and ai_pos == goal_pos):
            elapsed = int(time.time() - start_time)
            iq_score = max(80, 150 - elapsed - (30 if used_hint else 0))
            winner = "You" if player_pos == goal_pos else "AI Bot"
            draw_text(f"{winner} Win!", 72, YELLOW, SCREEN_WIDTH // 2, 120, center=True)
            draw_text(f"Your IQ Score: {iq_score}", 54, GREEN, SCREEN_WIDTH // 2, 220, center=True)
            pygame.display.flip(); time.sleep(2)
            return
        # Hint gesture
        if open_palm:
            if hint_time is None:
                hint_time = time.time()
            elif time.time() - hint_time > 2:
                show_hint = not show_hint
                used_hint = True
                hint_time = None
        else:
            hint_time = None
        # Exit gesture
        if time.time() - game_start_time > 3:
            if fist:
                if exit_time is None: exit_time = time.time()
                hold = time.time() - exit_time
                draw_exit_symbol((SCREEN_WIDTH//2, SCREEN_HEIGHT-100), hold, 2)
                if hold > 2: pygame.quit(); sys.exit()
            else: exit_time = None
        draw_camera_feed(frame)
        pygame.display.flip(); clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()

# --- Placeholder for Shooter, Pong, Breakout, Fruit Catcher with AI/Hint/IQ ---
def shooter_game(level=1, ai_mode=False): maze_game(level, ai_mode)  # Replace with full shooter logic
def pong_game(level=1, ai_mode=False): maze_game(level, ai_mode)  # Replace with full pong logic
def breakout_game(level=1, ai_mode=False): maze_game(level, ai_mode)  # Replace with full breakout logic
def fruit_catcher_game(level=1, ai_mode=False): maze_game(level, ai_mode)  # Replace with full fruit_catcher logic

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        introduction_screen()
        mode = mode_selection()  # 0: Single Player, 1: AI Bot
        selected = gesture_menu()
        if selected == 5:
            iq_analyzer()
        else:
            level = level_selection(MENU_ITEMS[selected])
            if selected == 0: shooter_game(level, ai_mode=(mode==1))
            elif selected == 1: pong_game(level, ai_mode=(mode==1))
            elif selected == 2: breakout_game(level, ai_mode=(mode==1))
            elif selected == 3: maze_game(level, ai_mode=(mode==1))
            elif selected == 4: fruit_catcher_game(level, ai_mode=(mode==1))
