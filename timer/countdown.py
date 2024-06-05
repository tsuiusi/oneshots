import tkinter as tk
from datetime import datetime, timedelta

def update_time():
    current_time = datetime.now()
    midnight = datetime.combine(current_time.date() + timedelta(days=1), datetime.min.time())
    time_diff = midnight - current_time
    intervals = time_diff.seconds // 900  # 900 seconds = 15 minutes
    
    time_label.config(text=f"you have {intervals}/96 turns left.\nlet's fucking go.")
    
    # Schedule the next update at midnight
    next_update = midnight + timedelta(seconds=1)
    time_diff = next_update - datetime.now()
    milliseconds = time_diff.total_seconds() * 1000
    window.after(int(milliseconds), update_time)

# Create the main window
window = tk.Tk()
window.title("Turns")
window.attributes("-topmost", True)

# Create a label to display the time
time_label = tk.Label(window, font=("Helvetica", 24, 'bold'))
time_label.pack(pady=20)

# Call the update_time function to start the clock
update_time()

# Start the Tkinter event loop
window.mainloop()
