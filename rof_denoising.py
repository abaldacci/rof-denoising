import cv2
import numpy as np


def gradient(u, u0, tau):
    du = cv2.Laplacian(u, cv2.CV_64F, ksize=3)
    return tau * (u - u0) - 2. * du


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

# denoised image
denoised_image = noisy_image.copy()
step = .01
u0 = noisy_image
u  = denoised_image
tau = 0.8
eps = 0.001
maxiters = 10000
for i in range(0, maxiters):
    g = gradient(u, u0, tau)
    gnorm = np.linalg.norm(g)
    print('- iter {}: gnorm {}'.format(i, gnorm))
    if gnorm < eps:
        break
    u = u - step * g

denoised_image   = u
difference_image = denoised_image - noisy_image

# convert back to uint8 for visualization
original_image   = np.clip(original_image   * 255., 0, 255).astype(np.uint8)
noisy_image      = np.clip(noisy_image      * 255., 0, 255).astype(np.uint8)
denoised_image   = np.clip(denoised_image   * 255., 0, 255).astype(np.uint8)
difference_image = np.clip(difference_image * 255., 0, 255).astype(np.uint8)

cv2.imshow('original image'  , original_image  )
cv2.imshow('noisy image'     , noisy_image     )
cv2.imshow('denoised image'  , denoised_image  )
cv2.imshow('difference image', difference_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
