import cv2
import numpy as np

# Read the image to be clustered
img = cv2.imread('output/example.jpg')

# Gives a new shape to an array without changing its data.
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters (K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8 and make original image
center = np.uint8(center)
res = center[label.flatten()]
result = res.reshape(img.shape)

# Show final results
cv2.imshow("result", np.hstack([img, result]))
cv2.imwrite('output/exampleclustering.jpg', np.hstack([img, result]))
cv2.waitKey(0)
cv2.destroyAllWindows()