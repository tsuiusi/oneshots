import numpy as np
import cv2

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
        self.avfactor = 0.5
        self.alignfactor = 1
        self.centerfactor = 1
        self.flock = 25  

    def alignment(self, boid):
        self.x_vel, boid.x_vel += self.alignfactor * np.mean(self.x_vel, boid.x_vel)
        self.y_vel, boid.y_vel += self.alignfactor * np.mean(self.y_vel, boid.y_vel)

    def fov(self, boids):
        domain = []
        for i in boids:
            if euclidean(i, self) <= self.radius:
                domain.append(i)
        return domain

    def avoidance(self, i):
        # not exactly because it does have to align, but would sufficient alignment solve avoidance
        self.x_vel, i.x_vel += self.alignfactor * i.x_vel, self.alignfactor * self.x_vel
        self.y_vel, i.y_vel += self.alignfactor * i.y_vel, self.alignfactor * self.y_vel

    def centering(self, boids):
        xs = []
        ys = []
        for i in boids:
            if euclidean(self, i) < self.radius:
                xs.append(i.x)
                ys.append(i.y)

        center = np.mean(xs), np.mean(ys)

        self.x_vel += self.centerfactor * (np.mean(xs) - self.x)
        self.y_vel += self.centerfactor * (np.mean(ys) - self.y)         

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
        # Centering
        boid.centering(boids)
        
        if euclidean(a, i) < a.radius:
            # Avoidance
            boid.avoidance(i)
            
            # Alignment 
            boid.alignment(i)

def main(n):
    boids = [boid() for _ in range(n)]

    while True:
        for i in range(n):
            apply_rules(boids[n], boids[:n, n:]) # i forgot how to do this

        if cv2.waitKey(1) = ord('q'):
            break

    cv2.destroyAllWindows()

if __init__ == "__main__":
    main()






   

    




    

