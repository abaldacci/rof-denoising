import cv2
import numpy as np


def forward_differences_newmann(x):
    res = np.zeros((x.shape[0],x.shape[1],2))
    res[:-1,:,0] = x[1:,:] - x[:-1,:]
    res[:,:-1,1] = x[:,1:] - x[:,:-1]
    return res


def forward_differences_newmann_conj(y):
    res        = np.zeros((y.shape[0],y.shape[1]))
    res[:-1,:] = -y[:-1,:,0]
    res[1:,:]  = res[1:,:] + y[:-1,:,0]
    res[:,:-1] = res[:,:-1] - y[:,:-1,1]
    res[:,1:]  = res[:,1:] + y[:,:-1,1]
    return res


def prox_project(clambda, z):
    nrm = np.sqrt(z[:,:,0]**2 + z[:,:,1]**2)
    fact = np.minimum(clambda, nrm)
    fact = np.divide(fact,nrm, out=np.zeros_like(fact), where=nrm!=0)

    y = np.zeros(z.shape)
    y[:,:,0] = np.multiply(z[:,:,0],fact)
    y[:,:,1] = np.multiply(z[:,:,1],fact)
    return y


def ROF_value(f,x,y,clambda):
    r""" Compute the ROF cost functional
    Parameters
    ----------
    f : numpy array
        Noisy input image
    x : numpy array
        Primal variable value
    y : numpy array
        Dual variable value
    clambda : float
        Tickonov regularization parameter
    """
    a = np.linalg.norm((f-x).flatten())**2/2
    b = np.sum(np.sqrt(np.sum(y**2,axis=2)).flatten())
    return a+clambda*b


def forward_backward_ROF(image, clambda, tau, iters=100):
    r""" Dual ROF solver using Forward-Backward Splitting
    Parameters
    ----------
    image : numpy array
        The noisy image we are processing
    clambda : float
        The non-negative weight in the optimization problem
    tau : float
        Parameter of the proximal operator
    iters : int
        Number of iterations allowed
    """
    print("2D Dual ROF solver using Forward-Backward Splitting")

    tau = tau / 8
    y = forward_differences_newmann(image)
    x = image.copy()

    vallog = np.zeros(iters)

    for i in range(iters):
        gradg = forward_differences_newmann(forward_differences_newmann_conj(y) - image)
        y = prox_project(clambda, y-tau*gradg)
        x = image-forward_differences_newmann_conj(y) #Retrieve primal value
        vallog[i] = ROF_value(image, x, forward_differences_newmann(x), clambda)
        print('- iter {}: rofvalue {}'.format(i, vallog[i]))

    return x


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
clambda = 0.1
tau = 0.5
denoised_image   = forward_backward_ROF(noisy_image, clambda, tau, 10000)
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
