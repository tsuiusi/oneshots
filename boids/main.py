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
    
    def avoidance(self):
        pass

    def alignment(self, another):
        v1 = np.sqrt(self.x_vel^2 + self.y_vel^2)
        v2 = np.sqrt(another.x_vel^2 + self,y_vel^2)
        


    def centering(self):
        pass

    def euclidean(self, another):
        # application of handshake problem so it doesn't go O(N^2)
        x = self.x - another.x
        y = self.y - another.y

        distance = np.sqrt(x^2 + y^2)

        return distance

    def fov(self):
        pass
    

