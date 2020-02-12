# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 05:31:42 2019

@author: KUSHAL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:48:10 2019
@author: crjones4
"""
import numpy as np
import cv2
import time
import scipy.ndimage as ndimg
from sklearn.cluster import KMeans
import argparse
import matplotlib.pyplot as plt

NROT = 6
NPER = 8
NFILT = NROT*NPER
FILTSIZE = 49
NCLUSTERS = 4
TEXELSIZE = 4
pathName = "C:\\Users\\Kushal Patel\\Desktop\\Courses\\Computer Vision\\Homework4"
fileName = "aerial-houses"
#fileName = "texture"
#fileName = "selfie"
###############################################################################
# This function will compute and return the Leung-Malik filters
# the filters are in a 3D array of floats, F(FILTSIZE, FILTSIZE, NFILT)
def makeLMfilters():
    def gaussian1d(sigma, mean, x, ord):
        x = np.array(x)
        x_ = x - mean
        var = sigma**2
    
        # Gaussian Function
        g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))
    
        if ord == 0:
            g = g1
            return g
        elif ord == 1:
            g = -g1*((x_)/(var))
            return g
        else:
            g = g1*(((x_*x_) - var)/(var**2))
            return g
    
    def gaussian2d(sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        return g
    
    def log2d(sup, scales):
        var = scales * scales
        shape = (sup,sup)
        n,m = [(i - 1)/2 for i in shape]
        x,y = np.ogrid[-m:m+1,-n:n+1]
        g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
        h = g*((x*x + y*y) - var)/(var**2)
        return h
    
    def makefilter(scale, phasex, phasey, pts, sup):
    
        gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
        gy = gaussian1d(scale,   0, pts[1,...], phasey)
    
        image = gx*gy
    
        image = np.reshape(image,(sup,sup))
        return image
    

    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F

###############################################################################
def saveFilters(img):
    (height, width, depth) = img.shape
    count = 0
    for row in range(NPER):
        for col in range(NROT):
            tempImg = img[:, :, count]
            filename = "Filters\\LM_" + str(row) + "_" + str(col)
            normedFilter = normImg(tempImg)
            saveImage(normedFilter, filename)
            count = count + 1
    return
###############################################################################
# this function will apply the filter bank in the 3D array filt to
# the inputImg; the result is an array of results res(height, width, NFILT)
def applyLMfilters(inputImg, filt):
    norm_img = inputImage
    norm_img = cv2.normalize(inputImg, norm_img, 0.0, 127.0, cv2.NORM_MINMAX)
    norm_img = norm_img + 128.0
    row = norm_img.shape[0]
    col = norm_img.shape[1]
    num_filt = len(filt[0,0,:])
    res = np.zeros((row,col,num_filt))
    for i in range(num_filt):
        curr_filter = filt[:,:,i]
        curr_res = cv2.filter2D(norm_img,-1, curr_filter)
        res[:,:,i] = curr_res
    return res
###############################################################################
def normImg(img):
    tempImg = np.zeros_like(img)
    tempImg = (cv2.normalize(img, tempImg, 0.0, 127.0, cv2.NORM_MINMAX))
    res = (tempImg+128.0).astype(np.uint8)
    return res
###############################################################################
def makeMosaic(img):
    (height, width, depth) = img.shape
    res = np.zeros((height*8, width*6), np.float64)
    count = 0
    for row in range(8):
        for col in range(6):
            res[row*height:(row+1)*height, col*width:(col+1)*width] = \
            normImg(img[:, :, count])
            count = count + 1
    return res
###############################################################################
def saveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)
    return
###############################################################################
# this function will take a 3D array of filter bank responses and form texels
# by combining the feature vectors in nonoverlapping squares of size sz
# the result newR is an array of floats the same size as R, but within
# texels all feature vectors are identical
def formTexels(R, sz):
    h, w, c = R.shape
    newR = np.zeros((h,w,NFILT))
    for filtr_ind in range(NFILT):
        b = R[:,:,filtr_ind]
        c = newR[:,:,filtr_ind]
        for i in range (0,h,sz):
            if i + sz == h:
                i = h - sz
            for j in range (0,w,sz):
                if j + sz == w:
                    j = j - sz
                window = b[i:i + sz, j:j + sz]
                mean = np.mean(window)
                c[i:i + sz, j:j + sz] = mean
        newR[0:h, 0:w, filtr_ind] = c
    return newR
###############################################################################
# this function will take an image-sized collection of filter bank responses
# and use the KMeans algorithm to find the best segments
# it returns a pseucolor rendition of the original image, where each color
# corresponds to a separate cluster (a separate type of texture)
    
def segmentKMeans(R, nclus):
    h, w, d = R.shape
    R = np.reshape(R,(h*w,d))
    clusters = KMeans(n_clusters = nclus, random_state=0).fit(R)
    img = np.reshape(clusters.labels_, (h, w))
    img = (img/3)*255
    img = np.array(img, dtype = np.uint8)
    pcolor = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
    return pcolor

###############################################################################
# This code sets the pathname from a command line option
# add the following as a command line option: --image_path="C:\\Data\\"
# replace C:\\Data with the proper path on your system
# Do NOT change this code – it’s used for grading and you WILL lose points!!!!
    
parser = argparse.ArgumentParser();
parser.add_argument('--image_path', required=True, help='Absolute path of the image to be used.');
if __name__ == '__main__':
    args = parser.parse_args();
    pathName = args.image_path;
    print('IMAGE PATH: ', pathName);
currTime = time.time()
# Call the make filter function
F = makeLMfilters()
saveFilters(F)
saveImage(makeMosaic(F), "allFilters")
# load an image
inputImage = cv2.cvtColor(cv2.imread(pathName + fileName + ".png"), cv2.COLOR_BGR2GRAY)
# find filter responses
rawR = applyLMfilters(inputImage, F)
if (True):#doMakeTexels):
    R = formTexels(rawR, TEXELSIZE)
else:
    R = rawR
    
# try segmenting
pcolor = segmentKMeans(R, NCLUSTERS)
plt.figure()
plt.imshow(pcolor)
saveImage(pcolor, fileName+"_Seg_"+str(NCLUSTERS))
elapsedTime = time.time() - currTime
print("Completed; elapsed time = ", elapsedTime)