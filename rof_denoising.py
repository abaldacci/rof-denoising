import cv2
import numpy as np

# load test image
original_image = cv2.imread('lena_color.png', cv2.IMREAD_GRAYSCALE)
original_image = original_image.astype(np.float32)
original_image /= 255.

# add gaussian noise
mean  = 0.0
var   = 0.01
sigma = var**0.5
gaussian_noise = np.random.normal(mean, sigma, original_image.shape)
gaussian_noise = gaussian_noise.reshape(original_image.shape)
noisy_image    = original_image + gaussian_noise


# convert back to uint8 for visualization
original_image = np.clip(original_image * 255., 0, 255).astype(np.uint8)
noisy_image    = np.clip(noisy_image * 255., 0, 255).astype(np.uint8)

cv2.imshow('original image', original_image)
cv2.imshow('noisy image'   , noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
