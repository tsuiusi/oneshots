import numpy as np

class boid():
    def __init__(self):
        # Coordinates
        self.x = 0
        self.y = 0

        # Velocity
        self.x_vel = 1
        self.y_vel = 1

        # Parameters
        self.radius = 1
        self.avfactor = 1
        self.matchfactor = 1
        self.flock = 25
    
def avoidance(a, neighbors): # b is a list?
    close_x = 0
    close_y = 0
    for i in neighbors:
        close_x += a.x - i.x
        close_y += a.y - i.y

    a.x_vel += close_x * a.avfactor
    a.y_vel += close_y * a.avfactor

def alignment():
     

def fov(self):
    pass

def centering(self):
    pass

def euclidean(a, b):
    # application of handshake problem so it doesn't go O(N^2)
    x = a.x - b.x
    y = a.y - b.y

    return np.sqrt(x^2 + y^2)

def apply_rules(a, boids):
    tot_x = 0
    tot_y = 0
    n = 0
    for i in boids: 
        if euclidean(a, i) < a.radius:
            # Alignment 
            n += 1
            tot_x += i.x_vel
            tot_y += i.y_vel

            # Avoidance
            boid.avoidance(i)

            # Centering
            centering(i)


   
    # Alignment
    a.x_vel += tot_x//n * a.matchfactor
    a.y_vel += tot_y//n * a.matchfactor
    




    

