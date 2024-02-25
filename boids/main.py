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
        self.alignfactor = 1
        self.centerfactor = 1
        self.flock = 25  

    def alignment(self, boid):
        self.x_vel += self.alignfactor * np.mean(self.x_vel, boid.x_vel)
        self.y_vel += self.alignfactor * np.mean(self.y_vel, boid.y_vel)

    def fov(self, boids):
        domain = []
        for i in boids:
            if euclidean(i, self) <= self.radius:
                domain.append(i)
        return domain

    def avoidance(self, i):
        # not exactly because it does have to align, but would sufficient alignment solve avoidance
        self.x_vel += self.alignfactor * np.mean(self.x_vel, boid.x_vel)
        self.y_vel += self.alignfactor * np.mean(self.y_vel, boid.y_vel)

    def centering(self, boids):
        # ??? 
        self.x -= self.centerfactor * (self.x - boid.x)
        self.y -= self.centerfactor * (self.y - boid.y)
         

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
            boid.alignment(i)  

            # Avoidance
            boid.avoidance(i)

            # Centering
            boid.centering(i)


   

    




    

