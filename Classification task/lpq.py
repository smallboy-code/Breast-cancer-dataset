
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


def lpq(img, winSize=9, freqestim=1, mode='im'):
    rho = 0.90

    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    '''
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)
    '''

    convmode = 'valid'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    # Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    # Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)

    # Switch format to uint8 if LPQ code np.image is required as output
    if mode == 'im':
        LPQdesc = np.uint8(LPQdesc)

    # Histogram if needed
    if mode == 'nh' or mode == 'h':
        LPQdesc = np.histogram(LPQdesc.flatten(), range(256))[0]

    # Normalize histogram if needed
    if mode == 'nh':
        LPQdesc = LPQdesc / LPQdesc.sum()

    print(LPQdesc)

    plt.figure()
    plt.imshow(LPQdesc,cmap='gray')
    plt.show()
    # return LPQdesc

path = r'./18_02_B_VIBRANT_36.jpg'
image = Image.open(path).convert('L')
lpq(image)
# path = r"E:\Data\Mydataset"
# for type in os.listdir(path):
#     type_path = os.path.join(path, type)
#     for patient in os.listdir(type_path):
#         patient_path = os.path.join(type_path, patient)
#         for jpg in os.listdir(patient_path):
#             image_path = os.path.join(patient_path, jpg)
#             image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             new_path = os.path.join('E:\Data\Mydataset/all\crop\LPQ/test', type)
#             if not os.path.exists(new_path):
#                 os.makedirs(new_path)
#             LPQdesc = lpq(gray)
#             cv2.imencode('.jpg', LPQdesc)[1].tofile(os.path.join(new_path, jpg))
# print('done')
# img = ndimage.imread('image1.jpg')
# print(gray)

