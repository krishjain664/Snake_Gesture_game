import pygame, sys, random, threading
from pygame.math import Vector2
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# ------------------- Shared Gesture Dictionary -------------------
shared_gesture = {"direction": None}
lock = threading.Lock()
running = True  # Shared flag to stop threads

# ------------------- Snake Classes -------------------
class Snake:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.direction = Vector2(1, 0)
        self.new_block = False
    
    def draw_snake(self):
        for block in self.body:
            block_rect = pygame.Rect(int(block.x * cell_size), int(block.y * cell_size), cell_size, cell_size)
            pygame.draw.rect(screen, (170, 140, 15), block_rect)

    def snake_motion(self):
        if self.new_block:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

class Fruit:
    def __init__(self):
        self.x = random.randint(0, cell_set - 1)
        self.y = random.randint(0, cell_set - 1)
        self.pos = Vector2(self.x, self.y)

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(screen, (126, 164, 114), fruit_rect)

    def randomize(self):
        self.x = random.randint(0, cell_set - 1)
        self.y = random.randint(0, cell_set - 1)
        self.pos = Vector2(self.x, self.y)

class MAIN:
    def __init__(self):
        self.snake = Snake()
        self.fruit = Fruit()

    def update(self):
        self.snake.snake_motion()
        self.check_collision()
        self.check_fail()
        self.update_direction_from_gesture()

    def draw_elements(self):
        self.fruit.draw_fruit()
        self.snake.draw_snake()

    def check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_set or not 0 <= self.snake.body[0].y < cell_set:
            self.game_over()
        for block in self.snake.body[1:]:
            if block == self.snake.body[0]:
                self.game_over()

    def game_over(self):
        global running
        running = False
        pygame.quit()
        sys.exit()

    def update_direction_from_gesture(self):
        with lock:
            gesture = shared_gesture["direction"]
        if gesture == "UP" and self.snake.direction.y != 1:
            self.snake.direction = Vector2(0, -1)
        elif gesture == "DOWN" and self.snake.direction.y != -1:
            self.snake.direction = Vector2(0, 1)
        elif gesture == "LEFT" and self.snake.direction.x != 1:
            self.snake.direction = Vector2(-1, 0)
        elif gesture == "RIGHT" and self.snake.direction.x != -1:
            self.snake.direction = Vector2(1, 0)

# ------------------- Initialize Pygame -------------------
pygame.init()
cell_size = 40
cell_set = 20
screen = pygame.display.set_mode((cell_size * cell_set, cell_size * cell_set))
clock = pygame.time.Clock()
main_game = MAIN()
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 400)  

def gesture_detection():
    global running
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    width, height = 80,100
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        running = False
        return
    cam.set(3, width)
    cam.set(4, height)

    while running:
        success, img_rgb = cam.read()
        if not success:
            continue

        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        hands_detected = hands.process(img_rgb)
        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

        if hands_detected.multi_hand_landmarks:
            hand_landmarks = hands_detected.multi_hand_landmarks[0]  # Use first hand
            landmarks = hand_landmarks.landmark
            # Getting wrist and index tip
            wrist = landmarks[0]
            index_tip = landmarks[8]
            if index_tip.y < wrist.y - 0.1:
                gesture = "UP"
            elif index_tip.y > wrist.y + 0.1:
                gesture = "DOWN"
            elif index_tip.x < wrist.x - 0.1:
                gesture = "RIGHT"
            elif index_tip.x > wrist.x + 0.1:
                gesture = "LEFT"
            else:
                gesture = None

            if gesture:
                with lock:
                    shared_gesture["direction"] = gesture

        cv.imshow("Webcam", cv.flip(img_rgb, 1))
        if cv.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cam.release()
    cv.destroyAllWindows()

def game_loop():
    global running
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == SCREEN_UPDATE:
                main_game.update()

        screen.fill((175, 215, 70))
        main_game.draw_elements()
        pygame.display.update()
        clock.tick(20)

gesture_thread = threading.Thread(target=gesture_detection, daemon=True)
game_thread = threading.Thread(target=game_loop, daemon=True)

gesture_thread.start()
game_thread.start()
gesture_thread.join()
game_thread.join()
