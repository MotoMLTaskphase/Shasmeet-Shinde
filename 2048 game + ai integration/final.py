import pygame
from ai import Tile, move_tiles, end_move, generate_tiles, draw


def evaluate_board(tiles):
    empty_tiles = len([key for key, value in tiles.items() if value is None])
    max_tile = max(int(key[0]) for key in tiles.keys())
    return empty_tiles + max_tile


def simulate_move(direction, tiles, window, clock):
    simulated_tiles = {key: value for key, value in tiles.items()}
    move_tiles(window, simulated_tiles, clock, direction)
    return evaluate_board(simulated_tiles)


def ai_make_move(tiles, window, clock):
    directions = ["up", "down", "left", "right"]
    best_direction = None
    best_score = float('-inf')
    for direction in directions:
        score = simulate_move(direction, tiles, window, clock)
        if score > best_score:
            best_score = score
            best_direction = direction


    if best_direction:
        move_tiles(window, tiles, clock, best_direction)
        end_move(tiles)



def ai_play_game():
    pygame.init()
    window = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    tiles = generate_tiles()

    while True:
        draw(window, tiles)
        ai_make_move(tiles, window, clock)


        if end_move(tiles) == "lost":
            print("Game Over")
            break

        clock.tick(30)

if __name__ == "__main__":
    ai_play_game()