import curses
import random
import time

# Initialize characters to use
matrix_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*"

def normal_int():
    # Generate a normally distributed random number centered at 0
    value = int(random.gauss(0, 0.7))
    return value

def matrix_screen(stdscr):
    curses.curs_set(0)
    height, width = stdscr.getmaxyx()
    
    # Initialize raindrops
    raindrops = []
    
    while True:
        stdscr.clear()

        # Spawn new raindrops
        if len(raindrops) < width // 2 and random.random() < 0.2:
            x = random.randint(0, width - 1)
            if not any(drop['x'][-1] == x for drop in raindrops):
                raindrops.append({
                    'x': [x],
                    'y': 0,
                    'length': random.randint(5, 15),
                })
        
        # Update and draw raindrops
        new_raindrops = []
        for drop in raindrops:
            # Random walk
            new_x = max(0, min(width - 1, drop['x'][-1] + normal_int()))
            drop['x'].append(new_x)
            
            # Keep only the last 'length' positions
            drop['x'] = drop['x'][-drop['length']:]
            
            # Draw the raindrop
            for i, x in enumerate(reversed(drop['x'])):
                y = drop['y'] - i
                if 0 <= y < height and 0 <= x < width:
                    char = random.choice(matrix_chars)
                    color = curses.color_pair(2) 
                    try:
                        stdscr.addch(y, x, char, color)
                    except curses.error:
                        pass  # Ignore errors from writing to bottom-right corner
            
            # Move the raindrop down
            drop['y'] += 1
            
            # Keep the raindrop if it's still on screen
            if drop['y'] - drop['length'] < height:
                new_raindrops.append(drop)
        
        raindrops = new_raindrops
        
        stdscr.refresh()
        time.sleep(0.05)

def main():
    stdscr = curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    
    try:
        matrix_screen(stdscr)
    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

if __name__ == "__main__":
    main()
