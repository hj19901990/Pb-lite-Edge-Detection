#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Rohitkrishna Nambiar (rohit517@terpmail.umd.edu)
Masters in Robotics
University of Maryland, College Park


Credits to Nitin J Sanket and Chahatdeep Singh for the template.
"""

# Code starts here:

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import argparse

def gaussianKernel2D(size=7, sigma=1):
	"""
	Create a 2D gaussian kernel given the kernel size and sigma value	
	"""	
	if (size % 2) != 1:
		return None

	sideWidth = (size - 1) / 2
	kernel = np.zeros((size, size), dtype=np.float32)
	for x in range(-sideWidth, sideWidth):
		for y in range(-sideWidth, sideWidth):
			X = x + sideWidth
			Y = y + sideWidth
			G = math.exp(-(math.pow(x,2)+math.pow(y,2))/(2*math.pow(sigma,2)))
			kernel[X,Y] = G

	kernel = kernel/kernel.sum()	
	return kernel

def multivarKernal(size=7, mean=[0, 0], covar=np.array([[1, 0],[0, 1]])):
	"""
	"""
	if (size % 2) != 1:
		return None

	sideWidth = (size - 1) / 2
	mu_1 = mean[0]
	mu_2 = mean[1]
	sig_1 = covar[0,0]
	sig_2 = covar[1,1]
	kernel = np.zeros((size, size), dtype=np.float32)
	for x in range(-sideWidth, sideWidth):
		for y in range(-sideWidth, sideWidth):
			X = x + sideWidth
			Y = y + sideWidth
			G1 = math.exp(-(math.pow(x-mu_1,2))/(2*math.pow(sig_1,2)))
			G2 = math.exp(-(math.pow(y-mu_2,2))/(2*math.pow(sig_2,2)))
			kernel[X,Y] = G1*G2

	return kernel


def orientedDoGFilterBank(size=9, scales=[1], orientation=1):
	"""
	Creates a DoG filter bank
	"""
	index = 0

	# Initialize filter bank
	DoGfilterBank = np.zeros((size, size, len(scales)*orientation), dtype=np.float32)	

	# Create a sobel mask
	sobel_kernel = np.array([[-1, 0, 1],
					   		[-2, 0, 2],
					   		[-1, 0, 1]], dtype=np.float32)

	for scale in scales:

		# Create a gaussian kernel
		gaussian_kernel = gaussianKernel2D(size, scale)

		# Convolve gaussian with sobel
		DoG = cv2.filter2D(gaussian_kernel, -1, sobel_kernel)

		rotateAngle = 360.0/orientation

		for i in range(orientation):
			DoGfilterBank[:,:,index] = ndi.interpolation.rotate(DoG, -i*rotateAngle, reshape=False)
			index += 1

	return DoGfilterBank, index


def addOrientedFilters(filterBank, image, kernel, degree, rotateAngle, orientation, ind):	
	for i in range(degree):
		image = cv2.filter2D(image, -1, kernel)

	for i in range(orientation):		
		filterBank[:,:,ind] = ndi.interpolation.rotate(image, -i*rotateAngle, reshape=False)
		ind = ind + 1

	return filterBank, ind

def getLMfilterBank():
	maxSizeDoG = 37
	noOfFilter = 48
	filterIndex = 0
	# Initialize filter bank
	LMfilterBank = np.zeros((maxSizeDoG, maxSizeDoG, noOfFilter), dtype=np.float32)	

	# First and second order DoG at 3 scales and 6 orientations (36)
	# Create a sobel mask
	sobel_kernel = np.array([[-1, -2, -1],
					   		[0, 0, 0],
					   		[1, 2, 1]], dtype=np.float32)
	
	# Scale = 2, degree = 1
	covari = np.array([[2,0],[0,6]])
	mvGaussian = multivarKernal(37, covar=covari)
	
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													1,
													30.0,
													6,
													filterIndex)
	# Scale = 2, degree = 2
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													2,
													30.0,
													6,
													filterIndex)

	# Scale = sqrt(2), degree = 1
	covari = np.array([[np.sqrt(2),0],[0,3*np.sqrt(2)]])
	mvGaussian = multivarKernal(37, covar=covari)
	
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													1,
													30.0,
													6,
													filterIndex)
	# Scale = sqrt(2), degree = 2
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													2,
													30.0,
													6,
													filterIndex)


	# Scale = 1, degree = 1
	covari = np.array([[1,0],[0,3]])
	mvGaussian = multivarKernal(37, covar=covari)
	
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													1,
													30.0,
													6,
													filterIndex)
	# Scale = 1, degree = 2
	LMfilterBank, filterIndex = addOrientedFilters(LMfilterBank, 
													mvGaussian,
													sobel_kernel,
													2,
													30.0,
													6,
													filterIndex)

	## LOG Filters	(8)
	laplacian_kernel = np.array([[0, 1, 0],
					   		[1, -4, 1],
					   		[0, 1, 0]], dtype=np.float32)

	# Scale = 1
	gaussian_kernel = gaussianKernel2D(37, 1)
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = sqrt(2)
	gaussian_kernel = gaussianKernel2D(37, np.sqrt(2))
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 2
	gaussian_kernel = gaussianKernel2D(37, 2)
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 2*sqrt(2)
	gaussian_kernel = gaussianKernel2D(37, 2*np.sqrt(2))
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 3
	gaussian_kernel = gaussianKernel2D(37, 3)
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 3*sqrt(2)
	gaussian_kernel = gaussianKernel2D(37, 3*np.sqrt(2))
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 6
	gaussian_kernel = gaussianKernel2D(37, 6)
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)
	filterIndex += 1

	# Scale = 6*sqrt(2)	
	gaussian_kernel = gaussianKernel2D(37, 6*np.sqrt(2))
	LMfilterBank[:,:,filterIndex] = cv2.filter2D(gaussian_kernel, -1, laplacian_kernel)	
	filterIndex += 1

	## Gaussians (4) ##
	# Scale = 1
	LMfilterBank[:,:,filterIndex] = gaussianKernel2D(37, 1)
	filterIndex += 1

	# Scale = sqrt(2)
	LMfilterBank[:,:,filterIndex] = gaussianKernel2D(37, np.sqrt(2))
	filterIndex += 1

	# Scale = 2
	LMfilterBank[:,:,filterIndex] = gaussianKernel2D(37, 2)
	filterIndex += 1

	# Scale = 2*sqrt(2)
	LMfilterBank[:,:,filterIndex] = gaussianKernel2D(37, 2*np.sqrt(2))
	filterIndex += 1	

	return LMfilterBank

def gaborKernel(ksize, wvlength=6, theta=0, offset=0, sigma=6, gamma=1):
	if (ksize % 2) != 1:
		return None

	sideWidth = (ksize - 1) / 2
	kernel = np.zeros((ksize, ksize), dtype=np.float32)
	a = -0.5/(math.pow(sigma,2))
	b = -(0.5*math.pow(gamma,2))/(math.pow(sigma,2))
	const = (2.0*math.pi)/wvlength
	ct = np.cos(theta)
	st = np.sin(theta)

	for x in range(-sideWidth, sideWidth):
		for y in range(-sideWidth, sideWidth):
			X = x + sideWidth
			Y = y + sideWidth
			x_d = x*ct + y*st
			y_d = -x*st + y*ct
			kernel[X,Y] = np.exp(math.pow(x_d,2)*a + math.pow(y_d,2)*b) * np.cos(const*x_d + offset)

	return kernel

def gaborFilterBank():
	maxSize = 37
	scales = [[4, 4], [6, 4], [8, 6], [10, 8], [12, 14]]
	orientation = 8
	index = 0

	# Initialize filter bank
	gaborfilterBank = np.zeros((maxSize, maxSize, len(scales)*orientation), dtype=np.float32)	

	rotateAngle = 3.14159/orientation

	for scale in scales:
		wv = scale[0]
		sig = scale[1]
		for i in range(orientation):
			gaborfilterBank[:,:,index] = gaborKernel(maxSize, 
													wvlength=wv,
													theta=i*rotateAngle,
													sigma=sig)
			index += 1

	return gaborfilterBank


def getHalfDiskMasks(radius, hdmOrientations):
	hdMasks = []
	for radii in radius:
		mask = np.zeros((radii*2+1, radii*2+1), dtype=np.float32)

		for i in range(radii):
			x = math.pow((i - radii),2)
			for j in range(radii*2+1):
				if x + math.pow((j - radii),2) < math.pow(radii,2):
					mask[i,j] = 1

		rotateAngle = 360.0/hdmOrientations
		for i in range(hdmOrientations):
			rotated = ndi.interpolation.rotate(mask, -i*rotateAngle, reshape=False)
			rotated[rotated > 1] = 1
			rotated[rotated < 0] = 0
			ret,rotated = cv2.threshold(rotated,0.5,1,cv2.THRESH_BINARY)

			# Rotated pair
			rotated_p = ndi.interpolation.rotate(mask, -i*rotateAngle-180, reshape=False)
			rotated_p[rotated_p > 1] = 1
			rotated_p[rotated_p < 0] = 0
			ret,rotated_p = cv2.threshold(rotated_p,0.5,1,cv2.THRESH_BINARY)

			hdMasks.append(rotated)
			hdMasks.append(rotated_p)

	return hdMasks


def getTextonMap(image, DoGfilterBank):
	shape = DoGfilterBank.shape

	noOfFilter = shape[2]

	# Initialize array to store filter response
	filter_response = np.zeros((image.shape[0], image.shape[1], noOfFilter))

	for i in range(noOfFilter):
		filter_response[:,:,i] = cv2.filter2D(image, -1, DoGfilterBank[:,:,i])

	reshaped = filter_response.reshape((-1,noOfFilter))
	reshaped = np.float32(reshaped)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 64
	ret,label,center=cv2.kmeans(reshaped,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	label = label.reshape(image.shape[0], image.shape[1])

	return label


def getBrightnessMap(image, clusters=16):
	reshaped = image.reshape((-1,1))
	reshaped = np.float32(reshaped)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(reshaped,clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	label = label.reshape(image.shape[0], image.shape[1])

	return label

def getColorMap(image, clusters=16):
	reshaped = image.reshape((-1,3))
	reshaped = np.float32(reshaped)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(reshaped,clusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	label = label.reshape(image.shape[0], image.shape[1])

	return label


def getGradient(image, hist_bins, halfDiskMasks, noScales, noOrientation):
	noOfFilter = noScales * noOrientation
	(row, col) = image.shape
	index = 0

	# Initialize array to store gradient
	gradient = np.zeros((row, col, noOfFilter), dtype=np.float32)
	small_array = np.full((row, col), np.spacing(np.single(1)), dtype=np.float32)

	for i in range(0, noOfFilter*2, 2):
		chi = np.zeros((row, col), dtype=np.float32)
		left_mask = halfDiskMasks[i]
		right_mask = halfDiskMasks[i+1]
		for k in range(hist_bins):
			temp = image.copy()
			temp[image == i] = 1.0
			temp[image != i] = 0.0			
			g_i = cv2.filter2D(temp, -1, left_mask)
			h_i = cv2.filter2D(temp, -1, right_mask)
			chi = chi + np.divide((g_i - h_i)**2, g_i + h_i + small_array)

		gradient[:,:,index] = 0.5*chi
		index +=1

	return gradient


def main():

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageName', default='7.jpg', help='Image name, Default:7.jpg')
	Parser.add_argument('--ShowPlots', default="True", help='String flag to show plot, Default:True')


	Args = Parser.parse_args()
	ImageName = Args.ImageName
	ShowPlots = Args.ShowPlots.lower() == 'true'

	ipSplit = ImageName.split('.')
	ipImageName = '../BSDS500/Images/' + ImageName

	image = cv2.imread(ipImageName, 0)
	image_color = cv2.imread(ipImageName)

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""		
	DoGfilterBank, index = orientedDoGFilterBank(15, [1, 2], 16)

	# Plot the filter bank	
	fig1 = plt.figure()
	for i in range(1, index+1):
		ax = fig1.add_subplot(2, 16, i)			
		plt.imshow(DoGfilterBank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig1.suptitle("DoG Filter Bank", fontsize=20)
	plt.savefig('../Output/DoG.png')	

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LMfilterBank = getLMfilterBank()
	
	# Plot the filter bank
	fig2 = plt.figure()
	for i in range(1, 49):
		ax = fig2.add_subplot(4, 12, i)
		plt.imshow(LMfilterBank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig2.suptitle("LM Filter Bank", fontsize=20)
	plt.savefig('../Output/LM.png')	

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gbFilterBank = gaborFilterBank()

	fig3 = plt.figure()
	for i in range(1, 41):
		ax = fig3.add_subplot(5, 8, i)
		plt.imshow(gbFilterBank[:,:,i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig3.suptitle("Gabor Filter Bank", fontsize=20)
	plt.savefig('../Output/Gabor.png')	

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	halfDiskMasks = getHalfDiskMasks([5, 10, 15], 8)

	fig4 = plt.figure()
	for i in range(1, len(halfDiskMasks)+1):
		ax = fig4.add_subplot(6, 8, i)
		plt.imshow(halfDiskMasks[i-1], interpolation='none', cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])

	fig4.suptitle("Half-Disk masks", fontsize=20)
	plt.savefig('../Output/HDMasks.png')

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	textonMap = getTextonMap(image, DoGfilterBank)

	fig5 = plt.figure()	
	plt.imshow(textonMap)
	tmFileName = "../Output/TextonMap_" + ipSplit[0] + ".png"
	fig5.suptitle("Texton Map", fontsize=20)
	plt.savefig(tmFileName)	

	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	## See above section


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	textonGradient = getGradient(textonMap, 64, halfDiskMasks, 3, 8)
	tgmean = textonGradient.mean(axis=2, dtype=np.float32)

	fig6 = plt.figure()	
	plt.imshow(tgmean, cmap='gray')
	tmFileName = "../Output/Tg_" + ipSplit[0] + ".png"
	fig6.suptitle("Texton Gradient", fontsize=20)
	plt.savefig(tmFileName)

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	brightnessMap = getBrightnessMap(image, 16)

	fig7 = plt.figure()	
	plt.imshow(brightnessMap)
	tmFileName = "../Output/BrightnessMap_" + ipSplit[0] + ".png"
	fig7.suptitle("Brightness Map", fontsize=20)
	plt.savefig(tmFileName)

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	brightnessGradient = getGradient(brightnessMap, 16, halfDiskMasks, 3, 8)
	bgmean = brightnessGradient.mean(axis=2, dtype=np.float32)

	fig8 = plt.figure()	
	plt.imshow(bgmean, cmap='gray')
	tmFileName = "../Output/Bg_" + ipSplit[0] + ".png"
	fig8.suptitle("Brightness Gradient", fontsize=20)
	plt.savefig(tmFileName)

	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	# We use RGB color image itself for clustering. Can try other
	# color spaces
	colorMap = getColorMap(image_color, 16)

	fig9 = plt.figure()	
	plt.imshow(colorMap)
	tmFileName = "../Output/ColorMap_" + ipSplit[0] + ".png"
	fig9.suptitle("Color Map", fontsize=20)
	plt.savefig(tmFileName)

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	colorGradient = getGradient(colorMap, 16, halfDiskMasks, 3, 8)
	cgmean = colorGradient.mean(axis=2, dtype=np.float32)

	fig10 = plt.figure()	
	plt.imshow(cgmean, cmap='gray')
	tmFileName = "../Output/Cg_" + ipSplit[0] + ".png"
	fig10.suptitle("Color Gradient", fontsize=20)
	plt.savefig(tmFileName)

	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	sobelPath = "../BSDS500/SobelBaseline/" + ipSplit[0] + ".png"
	sobel_base = cv2.imread(sobelPath, 0)

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	cannyPath = "../BSDS500/CannyBaseline/" + ipSplit[0] + ".png"
	canny_base = cv2.imread(cannyPath, 0)

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""		

	a = (tgmean + bgmean + cgmean)/3
	b = (0.5*canny_base + 0.5*sobel_base)
	pb = np.multiply(a, b)
	
	# plt.figure()
	# plt.imshow(pb_norm, cmap='gray')

	fig11 = plt.figure()	
	plt.imshow(pb, cmap='gray')
	tmFileName = "../Output/PbLite_" + ipSplit[0] + ".png"
	fig11.suptitle("Pb-lite Output", fontsize=20)
	plt.savefig(tmFileName)

	if ShowPlots:
		plt.show()
    
if __name__ == '__main__':
    main()
 


