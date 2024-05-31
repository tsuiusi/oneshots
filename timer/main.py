import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import time

class TimerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("10000 HOURS")
        
        # Load the background image using PIL
        self.background_image = Image.open("background.png")
        
        # Set the window size to match the image size
        self.master.geometry(f"{self.background_image.width}x{self.background_image.height}")
        
        # Convert the image to a PhotoImage
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        
        # Create the background label with the PhotoImage
        self.background_label = tk.Label(master, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.start_time = 0
        self.elapsed_time = 0
        self.running = False
        
        self.time_label = tk.Label(master, text="00:00:00", font=("Arial", 24, "bold"), bg="white", fg="black")
        self.time_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        
        self.percentage_label = tk.Label(master, text="0.00%", font=("Arial", 18, "bold"), bg="white", fg="black")
        self.percentage_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
        
        self.start_button = tk.Button(master, text="Start/Stop", font=("Arial", 14, "bold"), command=self.toggle_timer)
        self.start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
        
        self.load_elapsed_time()
        self.update_time()
        

    def toggle_timer(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            self.running = False
            self.elapsed_time += time.time() - self.start_time
    
    def update_time(self):
        if self.running:
            current_elapsed_time = self.elapsed_time + (time.time() - self.start_time)
        else:
            current_elapsed_time = self.elapsed_time
        self.display_time(current_elapsed_time)
        self.display_percentage(current_elapsed_time)
        self.master.after(1000, self.update_time)
    
    def display_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.time_label.config(text=time_string)
    
    def display_percentage(self, elapsed_time):
        percentage = (elapsed_time / 36000000) * 100
        percentage_string = f"{percentage:.2f}%"
        self.percentage_label.config(text=percentage_string)
    
    def load_elapsed_time(self):
        try:
            with open("timer_data.txt", "r") as file:
                self.elapsed_time = float(file.read())
                self.display_time(self.elapsed_time)
                self.display_percentage(self.elapsed_time)
        except FileNotFoundError:
            self.elapsed_time = 0
    
    def save_elapsed_time(self):
        with open("timer_data.txt", "w") as file:
            file.write(str(self.elapsed_time))

if __name__ == "__main__":
    root = tk.Tk()
    app = TimerApp(root)
    
    def on_close():
        app.save_elapsed_time()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
