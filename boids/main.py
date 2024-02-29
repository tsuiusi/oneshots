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


    def alignment(self, neighbors):
		# Calculate the average velocities of the neighbors
        avg_x_vel = np.mean([boid.x_vel for boid in neighbors])
        avg_y_vel = np.mean([boid.y_vel for boid in neighbors])
        print(f'Average X velocity: {avg_x_vel}')
        print(f'Average Y velocity: {avg_y_vel}')

        # Adjust the boid's velocity towards the average velocity of its neighbors
        # still doesn't work
        self.x_vel += np.round(self.alignfactor * (avg_x_vel - self.x_vel))
        self.y_vel += np.round(self.alignfactor * (avg_y_vel - self.y_vel))
            
        

    def fov(self, boids):
        domain = []
        for i in boids:
            if self.euclidean(i) <= self.radius:
                domain.append(i)
        return domain

    def avoidance(self, boids):
        # not exactly because it does have to align, but would sufficient alignment solve avoidance
        for i in boids:
            self.x_vel += self.avfactor * i.x_vel
            i.x_vel += self.avfactor * self.x_vel
            self.y_vel += self.avfactor * i.y_vel
            i.y_vel += self.avfactor * self.y_vel

    def centering(self, boids):   
        xs = [i.x for i in boids]
        ys = [i.y for i in boids]

        self.x_vel += self.centerfactor * (np.mean(xs) - self.x)
        self.y_vel += self.centerfactor * (np.mean(ys) - self.y)    

    def rotation(self):        
        return np.arctan(np.array([self.y_vel / self.x_vel]))
    

    def apply_rules(self, boids):
        tot_x = 0
        tot_y = 0
        n = 0
        for i in boids: 
			# Finding boids in radius
            boids = self.fov(boids)

            # Centering
            self.centering(boids)
		
			# Avoidance
            self.avoidance(boids)
			
			# Alignment 
            self.alignment(boids)


    def euclidean(self, b):
        # application of handshake problem so it doesn't go O(N^2)
        x = self.x - b.x
        y = self.y - b.y

        return np.sqrt(x**2 + y**2)



def main(n):
    boids = [boid() for _ in range(n)]

    while True:
        for i in range(n):
            boids[i].apply_rules(boids[:i] + boids[i+1:])

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if "__init__" == "__main__":
    main(5)

main(5)






   

    




    

