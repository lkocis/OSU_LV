import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50)) * 255

top_row = np.hstack((black, white))  
bottom_row = np.hstack((white, black))

final_image = np.vstack((top_row, bottom_row))  

plt.xticks(np.arange(0, 100, 20))  
plt.yticks(np.arange(0, 100, 20))
plt.gca().invert_yaxis()
plt.imshow(final_image, cmap='gray', vmin=0, vmax=255)
plt.show()