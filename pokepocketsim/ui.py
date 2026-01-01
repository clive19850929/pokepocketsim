import json
import pygame
from pathlib import Path

# ---------- 設定 ----------
BASE_DIR     = Path(__file__).resolve().parent
WIN_W, WIN_H = 960, 720
ASSETS       = BASE_DIR / "assets"
REPLAY_FILE  = BASE_DIR / "data" / "replay.json"
FPS          = 60
FONT_SIZE    = 18
LOG_LINES    = 5              # 画面に同時表示する最大行
# --------------------------

class CardSprite(pygame.sprite.Sprite):
    def __init__(self, card_id: str, pos):
        super().__init__()
        image_path = ASSETS / f"{card_id.lower()}.png"
        self.image = pygame.image.load(image_path).convert_alpha()
        self.rect  = self.image.get_rect(center=pos)

    def update(self, events):
        # ドラッグ＆ドロップ
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(e.pos):
                self.drag_offset = pygame.Vector2(self.rect.topleft) - e.pos
                self.dragging    = True
            elif e.type == pygame.MOUSEBUTTONUP:
                self.dragging = False
            elif e.type == pygame.MOUSEMOTION and getattr(self, "dragging", False):
                self.rect.topleft = e.pos + self.drag_offset

def main():
    # ----- Pygame 初期化 -----
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("poke-pocket-sim Viewer")

    font     = pygame.font.SysFont("sans-serif", FONT_SIZE)
    bg_board = pygame.image.load(ASSETS / "board.png").convert()

    # ----- リプレイ読み込み -----
    turns = json.load(REPLAY_FILE.open("r", encoding="utf-8"))
    turn_idx = 0

    def load_turn(t):
        sprites = pygame.sprite.Group()
        for c in t["board"]:
            sprites.add(CardSprite(c["id"], (c["x"], c["y"])))
        return sprites

    sprites = load_turn(turns[turn_idx])
    log_buf = turns[turn_idx]["log"]

    # ----- メインループ -----
    while True:
        events = [pygame.event.wait()] + pygame.event.get()  # 最低 1 イベント待機
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit(); return
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RIGHT and turn_idx < len(turns) - 1:
                    turn_idx += 1
                    sprites  = load_turn(turns[turn_idx])
                    log_buf  = turns[turn_idx]["log"]
                elif e.key == pygame.K_LEFT and turn_idx > 0:
                    turn_idx -= 1
                    sprites  = load_turn(turns[turn_idx])
                    log_buf  = turns[turn_idx]["log"]

        sprites.update(events)

        # ----- 描画 -----
        screen.blit(bg_board, (0, 0))
        sprites.draw(screen)

        # ログ表示（右上固定）
        y0 = 10
        for i, line in enumerate(log_buf[-LOG_LINES:][::-1]):
            txt = font.render(line, True, (255, 255, 255))
            screen.blit(txt, (WIN_W - txt.get_width() - 10, y0 + i * (FONT_SIZE + 2)))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
