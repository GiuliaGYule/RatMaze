import pygame
import random
import sys
from heapq import heappop, heappush

# Maze dimensions
width, height = 20, 20
cell_size = 20

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
brown = (165, 42, 42)
gray = ()

# Initialize Pygame
pygame.init()

# Set up the screen
screen_width = width * cell_size
screen_height = height * cell_size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Maze Generator")

# Clock to control the frame rate
clock = pygame.time.Clock()

# Function to generate the maze using recursive backtracking
def generate_maze():
    maze = [[1] * width for _ in range(height)]

    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height and maze[x][y] == 1

    def backtrack(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                maze[x + dx // 2][y + dy // 2] = 0
                maze[nx][ny] = 0
                backtrack(nx, ny)

    # Mark the starting point as a path
    maze[1][1] = 0
    backtrack(1, 1)

    # Mark the middle point as the end
    maze[width // 2][height // 2] = 2

    return maze

# Function to draw the maze
def draw_maze(maze):
    for i in range(width):
        for j in range(height):
            color = white if maze[i][j] == 0 else black if maze[i][j] == 1 else green
            pygame.draw.rect(screen, color, (i * cell_size, j * cell_size, cell_size, cell_size))

# Function to draw the mouse
def draw_mouse(x, y):
    mouse_color = brown
    pygame.draw.circle(screen, mouse_color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 2)

def draw_maze_with_highlight(maze, distances_center, distances_traveled, shortest_path):
    for i in range(width):
        for j in range(height):
            color = white if maze[i][j] == 0 else black if maze[i][j] == 1 else green

            # Check if the cell is in the shortest path
            if (i, j) in shortest_path:
                color = (255, 0, 0)  # Change the color to red for the shortest path

            pygame.draw.rect(screen, color, (i * cell_size, j * cell_size, cell_size, cell_size))

# Flood-fill algorithm to calculate distances
def flood_fill(maze, start):
    distances_center = [[sys.maxsize] * height for _ in range(width)]
    distances_traveled = [[sys.maxsize] * height for _ in range(width)]
    
    queue = [start]
    distances_center[start[0]][start[1]] = 0
    distances_traveled[start[0]][start[1]] = 0

    while queue:
        x, y = queue.pop(0)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[nx][ny] == 0:
                new_distance_center = distances_center[x][y] + 1
                new_distance_traveled = distances_traveled[x][y] + 1

                if new_distance_center < distances_center[nx][ny]:
                    distances_center[nx][ny] = new_distance_center
                    distances_traveled[nx][ny] = new_distance_traveled
                    queue.append((nx, ny))

    return distances_center, distances_traveled


# Find the next move based on the shortest distance, prioritizing the center
def find_next_move(weights, available_moves, goal):
    min_combined_weight = sys.maxsize
    next_move = None

    for move in available_moves:
        x, y = move
        distance_to_goal = abs(x - goal[0]) + abs(y - goal[1])
        combined_weight = weights[x][y] + distance_to_goal

        if combined_weight < min_combined_weight:
            min_combined_weight = combined_weight
            next_move = move

    return next_move

def reconstruct_path(distances_center, distances_traveled, start, end):
    x, y = end
    path = [(x, y)]

    while (x, y) != start:
        min_distance = sys.maxsize
        next_move = None

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                total_distance = distances_center[nx][ny] + distances_traveled[nx][ny]
                if total_distance < min_distance:
                    min_distance = total_distance
                    next_move = (nx, ny)

        if next_move:
            x, y = next_move
            path.append((x, y))

    return path

def dijkstra(maze, start, dest):
    rows, cols = len(maze), len(maze[0])
    visited = [[False] * cols for _ in range(rows)]
    distance = [[float('inf')] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]

    distance[start[0]][start[1]] = 0
    heap = [(0, start)]

    while heap:
        current_dist, current = heappop(heap)

        if current == dest:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current[0]][current[1]]
            return path[::-1]

        if visited[current[0]][current[1]]:
            continue

        visited[current[0]][current[1]] = True

        neighbors = get_neighbors(current[0], current[1], maze)
        for neighbor in neighbors:
            if not visited[neighbor[0]][neighbor[1]]:
                new_distance = distance[current[0]][current[1]] + 1
                if new_distance < distance[neighbor[0]][neighbor[1]]:
                    distance[neighbor[0]][neighbor[1]] = new_distance
                    parent[neighbor[0]][neighbor[1]] = current
                    heappush(heap, (new_distance, neighbor))

    return []

def get_neighbors(x, y, maze):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 3:
            neighbors.append((nx, ny))

    return neighbors

# Main function
def main():
    maze = generate_maze()
    mouse_x, mouse_y = 1, 1  # Initial position of the mouse

    maze[mouse_x][mouse_y] = 3

    running = True  # Flag to control the simulation

    visited_cells = []  # List to store the visited cells
    path = [(mouse_x, mouse_y)]  # Variable to store the mouse's path

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Set the flag to end the simulation
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False  # Set the flag to end the simulation

        screen.fill(white)
        draw_maze(maze)
        draw_mouse(mouse_x, mouse_y)

        distances_center, distances_traveled = flood_fill(maze, (mouse_x, mouse_y))
        center = (width // 2, height // 2)

        # Check if the mouse is right next to the end
        if (
            (mouse_x == center[0] and abs(mouse_y - center[1]) <= 1)
            or (mouse_y == center[1] and abs(mouse_x - center[0]) <= 1)
        ):
            running = False  # Mouse is right next to the end, end the simulation

            print("Visited Cells:")
            for cell in visited_cells:
                print(f"Cell: ({cell[0]}, {cell[1]})")

            # Print the maze, starting point, and destination point
            print("Maze:")
            for row in maze:
                print(row)

            print("Starting point:", (1, 1))
            print("Destination point:", (mouse_x, mouse_y))

            # Find the path from the starting point to the cell where the mouse ends
            shortest_path = dijkstra(maze, (1, 1), (mouse_x, mouse_y))
            print("Shortest Path using Dijkstra's Algorithm:")
            print(shortest_path)

             # Draw the maze with the shortest path highlighted
            draw_maze_with_highlight(maze, distances_center, distances_traveled, shortest_path)
        else:
            # Combine distances to determine the weight for each cell
            weights = [
                [
                    distances_center[i][j] + distances_traveled[i][j]
                    for j in range(height)
                ]
                for i in range(width)
            ]

            # Find available moves
            available_moves = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = mouse_x + dx, mouse_y + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and maze[nx][ny] == 0
                    and (nx, ny) not in path  # Ensure not revisiting the same cell
                ):
                    available_moves.append((nx, ny))

            if available_moves:
                # Choose the move with the lowest combined weight and distance to the goal
                next_move = find_next_move(weights, available_moves, center)

                if next_move is not None:
                    nx, ny = next_move
                    maze[nx][ny] = 3  # Update the maze to mark the visited cell
                    mouse_x, mouse_y = nx, ny

                    # Record the visited cell
                    visited_cells.append((nx, ny, weights[nx][ny]))

                    # Update the path
                    path.append((nx, ny))
            else:
                # Backtrack if there are no available moves
                if len(path) > 1:
                    prev_x, prev_y = path.pop()
                    mouse_x, mouse_y = prev_x, prev_y
                else:
                    running = False  # No alternative path, end the simulation

        pygame.display.flip()
        clock.tick(10)  # Adjust the speed of the mouse

    pygame.time.wait(5000)  # Wait for 5 seconds before closing (adjust as needed)
    pygame.quit()


if __name__ == "__main__":
    main()
