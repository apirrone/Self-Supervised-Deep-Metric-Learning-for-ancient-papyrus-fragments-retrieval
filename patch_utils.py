import os
import sys
import cv2
import random
import numpy as np
import collections
import re

def computeScore(backgroundScore, linesScore):
    # backgroundScore -> [0, 1] : 0 all background, 1 : no background
    # linesScore      -> [0, 1] : 0 no lines      , 1 : all pixels are part of lines
    #    in practice, when there is acceptable amount of line in a patch, the score is about 0.1 or 0.2

    return backgroundScore + linesScore

def colorsDist(color1, color2):
    assert len(color1) == len(color2)
    sum = 0
    for i in range(0, len(color1)):
        sum += ((color1[i]-color2[i])**2)
    return np.sqrt(sum)

def closestColor(color, colors):
    minDist = 1000000
    minDistColor = "white"
    for k, v in colors.items():
        dist = colorsDist(color, v)
        
        if dist < minDist:
            minDist = dist
            minDistColor = k

    return minDistColor

def extractBestBigPatch(im, minPatchSize, textMask=None, mask=None):
    if(im.shape[0] < minPatchSize or im.shape[1] < minPatchSize): # the image is too small
        smallestDimension = min(im.shape[0], im.shape[1])
        ratio = (minPatchSize+1)/smallestDimension
        newSize = (int(im.shape[0]*ratio), int(im.shape[1]*ratio))

        im = cv2.resize(im, newSize, interpolation = cv2.INTER_CUBIC)


    squareSize = min(im.shape[0], im.shape[1])

    if squareSize == im.shape[0]:
        posx = im.shape[1]//2
        y1 = 0
        y2 = squareSize-1
        x1 = posx-squareSize//2
        x2 = posx+squareSize//2
    else:
        posy = im.shape[0]//2
        x1 = 0
        x2 = squareSize-1
        y1 = posy-squareSize//2
        y2 = posy+squareSize//2
        
    patch = im[y1:y2, x1:x2]

    patch = cv2.resize(patch, (minPatchSize, minPatchSize), interpolation = cv2.INTER_CUBIC)

    return patch

# less long, 
def simpler_extractBestPatches(im, minPatchSize, nbPatches, textMask=None, mask=None):
    if(im.shape[0] < minPatchSize or im.shape[1] < minPatchSize): # the image is too small
        return [extractBestBigPatch(im, minPatchSize, textMask=textMask, mask=mask)]
    patches = []
    
    minIntensityThreshold = 20#50
    
    for i in range(0, im.shape[0] - minPatchSize, minPatchSize):
        for j in range(0, im.shape[1] - minPatchSize, minPatchSize):
            x1 = j
            x2 = j+minPatchSize
            y1 = i
            y2 = i+minPatchSize
            p = im[y1:y2, x1:x2]         
            if (mask is not None):
                maskPatch = np.array(mask[y1:y2, x1:x2])
     

                # exit()
                backgroundScore = ((maskPatch==1).sum()/(maskPatch.shape[0]*maskPatch.shape[1]))

                if (textMask is not None) :
                    textMaskPatch = textMask[y1:y2, x1:x2]

                    nbWhitePixels = (textMaskPatch>minIntensityThreshold).sum()
                    nbPixels = minPatchSize**2
                    linesScore = nbWhitePixels/nbPixels
                    # print(linesScore)
                    score = computeScore(backgroundScore, linesScore)
                else:
                    score = backgroundScore

            elif (textMask is not None) :
                textMaskPatch = textMask[y1:y2, x1:x2]
                
                nbWhitePixels = (textMaskPatch>minIntensityThreshold).sum()
                nbPixels = minPatchSize**2
                linesScore = nbWhitePixels/nbPixels
                
                score = linesScore
            else:
                print("ERROR : at least textMask or mask are needed")
                return []

            if score > 0.2:
                patches.append({"patch":p, "score":score, "pos":[x1, y1]})
    
    patches = sorted(patches, key = lambda i: i['score'], reverse=True)
    
    return_patches = []
    
    for p in patches:
        return_patches.append(p["patch"])

    return return_patches[:nbPatches]

def build_papyrus_mask(im, michigan=True):

    resize_factor = 10
    
    newSize = (im.shape[1]//resize_factor, im.shape[0]//resize_factor)
    tmpIm = cv2.resize(im, newSize, interpolation = cv2.INTER_CUBIC)

    mask = np.zeros((tmpIm.shape[0], tmpIm.shape[1])).astype(np.uint8)


    colors = {"white"        : [255, 255, 255], 
              "brown"        : [181, 131, 111],
              "black"        : [0, 0 ,0]}


    background_color = "white" if michigan == True else "black"
    
    for i in range(0, tmpIm.shape[0]):
        for j in range(0, tmpIm.shape[1]):
            color = tmpIm[i][j]
            if len(im.shape) == 3: # color image
                if closestColor(color, colors) != background_color:
                    mask[i][j] = 1
            else: # gray scale image
                if color > 5:
                    mask[i][j] = 1
                    
    mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation = cv2.INTER_CUBIC)
                
    return mask

def getCorrespondingMaskFileWithPath(f):
    fileName = f.split('/')[len(f.split('/'))-1]
    path = f[:-len(fileName)]
    return path+"/infered_"+fileName

def getGeshaemMask(fragment_file, resize=None):
    im = cv2.imread(fragment_file[:-len(".JPG")]+'_mask.JPG', cv2.IMREAD_GRAYSCALE)//255
    if resize is not None:
        im = cv2.resize(im, (0, 0), fx=resize, fy=resize)
    return im
    

def getPatchesFromImage(fragment_file, patch_size, nbChannels, max_patches_per_fragment, oneBigPatch=False, textMaskAvailable=False, michigan=True, geshaem=False, resize=None):
    imShape = (patch_size, patch_size, nbChannels)
    if nbChannels == 1:
        imShape = (patch_size, patch_size)
        
    fragment_textMask = None
    
    if textMaskAvailable:
        factor = 3.0303030303030303 # to go back to original size (infered is 33% of original size)
        fragment_textMask_file = getCorrespondingMaskFileWithPath(fragment_file)
        fragment_textMask = cv2.resize(cv2.imread(fragment_textMask_file), (0,0), fx=factor, fy=factor)
        if resize is not None:
            fragment_textMask = cv2.resize(fragment_textMask, (0, 0), fx=resize, fy=resize)

    if nbChannels == 3:
        fragment = cv2.imread(fragment_file)
    else:
        fragment = cv2.imread(fragment_file, cv2.IMREAD_GRAYSCALE)

        
    if resize is not None:
        fragment = cv2.resize(fragment, (0, 0), fx=resize, fy=resize)
        
    if geshaem:
        fragment_mask = getGeshaemMask(fragment_file)
    else:
        fragment_mask = build_papyrus_mask(fragment, michigan)

    
    if resize is not None:
        fragment_mask = cv2.resize(fragment_mask, (0, 0), fx=resize, fy=resize)

        
    if not oneBigPatch:
        fragment_patches = simpler_extractBestPatches(fragment, patch_size, max_patches_per_fragment, textMask=fragment_textMask, mask=fragment_mask)
    else:
        fragment_big_patch = extractBestBigPatch(fragment, patch_size, textMask=fragment_textMask, mask=fragment_mask)
        if len(fragment_big_patch) > 0:
            fragment_patches = [fragment_big_patch]
        else:
            fragment_patches = []


            
    for p in fragment_patches:
        assert p.shape == imShape, "you're in bad shape : "+str(p.shape)+" != "+str(imShape)
        
    return fragment_patches
