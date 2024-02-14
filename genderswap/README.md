# Gender swap filter
The title is self explanatory.

## Roadmap
1. Face landmark detection
2. Male/female detection
3. Male/female filter/diffusion model? 
4. Application and drawing the filter on top

## Dev notes
* From what I'm reading the rotation is done by anchoring certain points and getting the rotated points of the face (constants)
* Something called procrustes projection alignment
* Calculate the face's distances and landmarks and etc
* Get image parameters from the name (landmarks, direction, etc) and pass it through the add_filter function later


## Deadlines
| Date | Goal |
| -- | -- |
| 14/02/24 | working filter |
| 20/02/24 | wraparound crown | 

