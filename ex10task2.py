# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:59:51 2017

@author: andre
"""

# import packages
import numpy as np
import PIL
from PIL import Image

# import images
image1 = Image.open('CAT_black_nose.png')
image2 = Image.open('CAT_red_nose.png')
image3 = Image.open('CAT_mexican.png')

#%%

# convert to grayscale
def ImageHashingHex(image):
    image_grey = image.convert('L') # convert to gayscale
    image_grey_9x8 = image_grey.resize((9,8), Image.ANTIALIAS) # resize to 9 x 8 pixels
    image_grey_9x8_arrray = np.asarray(image_grey_9x8) # convert into aray of numbers
    true_false0 = np.zeros([image_grey_9x8_arrray.shape[0], image_grey_9x8_arrray.shape[1]-1])
    true_false = true_false0.astype(int) #  true/false matrix
    for i in range (0,image_grey_9x8_arrray.shape[0]):
        for j in range (0,image_grey_9x8_arrray.shape[1]-1):
            if image_grey_9x8_arrray[i,j] > image_grey_9x8_arrray[i,j+1]:
                true_false[i,j] = 1
    binary_list =list(true_false.flat) # matrix -> binary list
    binary_string = ''.join(map(str, binary_list))
    hex_string = hex(int(binary_string,2)) # binary list -> hash
    return  hex_string;

print(ImageHashingHex(image1))
print(ImageHashingHex(image2))
print(ImageHashingHex(image3))

#%%
#This is some additional code that calculates the similarity

def ImageHashing(image):
    image_grey = image.convert('L') # convert to gayscale
    image_grey_9x8 = image_grey.resize((9,8), Image.ANTIALIAS) # resize to 9 x 8 pixels
    image_grey_9x8_arrray = np.asarray(image_grey_9x8) # convert into aray of numbers
    # hashing
    true_false0 = np.zeros([image_grey_9x8_arrray.shape[0], image_grey_9x8_arrray.shape[1]-1])
    true_false = true_false0.astype(int)
    for i in range (0,image_grey_9x8_arrray.shape[0]):
        for j in range (0,image_grey_9x8_arrray.shape[1]-1):
            if image_grey_9x8_arrray[i,j] > image_grey_9x8_arrray[i,j+1]:
                true_false[i,j] = 1
    binary_list =list(true_false.flat)
    return  binary_list ;

#print(ImageHashing(image1))
#print(ImageHashing(image2))
print('similarity:',(1-np.sum(abs(np.subtract(ImageHashing(image1),ImageHashing(image2))))/len(ImageHashing(image1)))*100,'%')


