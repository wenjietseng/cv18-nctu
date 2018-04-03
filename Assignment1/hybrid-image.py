import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift 
import cv2
import math
import time
import sys

# Get the value of center in order to transfer to transfer to the coordinate where 
# origin is at the center of the image
def gaussianFilter(numRows, numCols, sigma, highPass=True):
    if(numRows % 2):
        centerI = int(numRows/2) + 1
    else:
        centerI = int(numRows/2)
    if(numCols % 2):
        centerJ = int(numCols/2) + 1
    else:
        centerJ = int(numCols/2)

    # Calculated by the definition of gaussian filter
    def getCoefficient(u, v):
        coefficient = math.exp(-1.0 * ((u - centerI)**2 + (v - centerJ)**2) / (2 * sigma**2))
        if(highPass):
            return 1 - coefficient
        else:
            return coefficient

    # Form the whole filter plane
    return numpy.array([[getCoefficient(i,j) for j in range(numCols)] for i in range(numRows)])


def filterDFT(imageMatrix, filterMatrix):
    shiftedDFT = fftshift(fft2(imageMatrix))

    # Just to see the FFT result
    cv2.imwrite("dft.jpg", numpy.real(shiftedDFT))

    filteredDFT = shiftedDFT * filterMatrix

    # To see the filtered FFT result
    cv2.imwrite("filtered-dft.jpg", numpy.real(filteredDFT))

    return ifft2(ifftshift(filteredDFT))

def lowPass(imageMatrix, sigma):
    height, width = imageMatrix.shape
    return filterDFT(imageMatrix, gaussianFilter(height, width, sigma, highPass = False))


def highPass(imageMatrix, sigma):
    height, width = imageMatrix.shape
    return filterDFT(imageMatrix, gaussianFilter(height, width, sigma))


def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
    highPassedImage = highPass(highFreqImg, sigmaHigh)
    lowPassedImage = lowPass(lowFreqImg, sigmaLow)

    return highPassedImage + lowPassedImage

def main():

    einstein = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    marilyn = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    hybrid = hybridImage(einstein, marilyn, 25, 10)
    cv2.imwrite("marilyn-einstein.png", numpy.real(hybrid))


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))