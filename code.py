# Digital Image Processing 
# Assignment A 2021

import numpy as np
import cv2
#import math
import random


def myConv2(A, B, param):
    # do 2 dimensional convolution here
    # param can be 'pad' or 'same'

    #Create an empty matrix 
    C = np.zeros_like(A)  

    
    # Flip the kernel
    B = np.flipud(np.fliplr(B))
    
    #Check for zero padding
    if param == 'same':
        pad = 0
        A = A
        
    elif param == 'pad':
        pad = 17 #9
        A = pad_image(A,pad)
       
    print(A) 
    print(A.shape)
 

    # Gather Shapes of Kernel 
    xKernShape = B.shape[1]
    yKernShape = B.shape[0]  
    
    for y in range(A.shape[0]):
         for x in range(A.shape[1]):
                 try:    
                     vert_start = y
                     vert_end = y + yKernShape
                     horiz_start = x 
                     horiz_end = x + xKernShape
                     s = B[:,:]*A[ vert_start:vert_end, horiz_start:horiz_end]     
                     C[ y, x] = np.sum(s)               
                 except:
                     break             
    return(C)
    
    

def myImNoise(A, param):
    # add noise according to the parameters
    # param must be at least 'gaussian' and 'saltandpepper'
    
    if param == 'gaussian':
        sigma = 2
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
        m = filter_size//2
        n = filter_size//2
        
        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = 2*np.pi*(sigma**2)
                x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
                gaussian_filter[x+m, y+n] = (1/x1)*x2
        
        im_filtered = np.zeros_like(A, dtype=np.float32)

        im_filtered[:, :] = myConv2(A[:, :], gaussian_filter,'pad')
           
        noise = im_filtered.astype(np.uint8)
        
        #noise = cv2.GaussianBlur(A,(9,9),0)
    elif param == 'saltandpepper':

       
        w , h = A.shape #Dimensions of image

        num_pixels = random.randint(400, 20000)
        for i in range(num_pixels):
            
            # Pick a random y coordinate
            y=random.randint(0, w - 1)
              
            # Pick a random x coordinate
            x=random.randint(0, h - 1)
              
            # Color that pixel to white
            A[y][x] = 255
              

        num_pixels = random.randint(400 , 20000)
        for i in range(num_pixels):
            
            # Pick a random y coordinate
            y=random.randint(0, w - 1)
              
            # Pick a random x coordinate
            x=random.randint(0, h - 1)
              
            # Color that pixel to black
            A[y][x] = 0
              
        return A
        #noise = myConv2(A,kernel, A.shape)
    return noise
 

    
    
def myImFilter(A, param):
    # fitler image A according to the parameters
    # param must be at least 'mean' and 'median'
    if param == 'mean':
        K = create_smooth_kernel(3)
        An = myConv2(A,K, 'same')
       
        #An = cv2.blur(A, (3,3))
        return An
    elif param == 'median':

        m, n = A.shape #rows  and columns of image
        An = np.zeros([m, n])
          
        for i in range(1, m-1):
            for j in range(1, n-1):
                md = [A[i-1, j-1],
                       A[i-1, j],
                       A[i-1, j + 1],
                       A[i, j-1],
                       A[i, j],
                       A[i, j + 1],
                       A[i + 1, j-1],
                       A[i + 1, j],
                       A[i + 1, j + 1]]
                  
                md = sorted(md)
                An[i, j]= md[4]
        #An = cv2.medianBlur(A,3)
        return An
  
        

#Create kernel             
def create_smooth_kernel(size):
    
    matrix = np.ones((size, size), dtype="float") * (1.0 / (size **2))

    return(matrix)

# Add zero padding         
def pad_image(A, size):
    pad = (size-1)/2 #stride is 1
    pad = int(pad)
    Apad = np.pad(A,pad_width=( (pad,pad), (pad,pad)), mode='constant', constant_values =0).astype(np.float32)

    return(Apad)



def main():
    #read image
    image = cv2.imread("kitty.jpg")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    #get image dimensions
    (h, w, d) = image.shape
    print("width={}, height={}, depth={}".format(w, h, d))
   
    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale image", gray)
    cv2.waitKey(0)
   
    
    #Add noise with gaussian param
    A = myImNoise(gray, 'gaussian') 
    cv2.imshow('Gaussian image',A) #show image
    cv2.waitKey(0) #wait for key press
    cv2.destroyAllWindows() #close image window
    
    #Add noise with saltandpepper param
    B = myImNoise(gray, 'saltandpepper')
    cv2.imshow('Saltandpepper image',B) #show image
    cv2.waitKey(0) #wait for key press
    cv2.destroyAllWindows() #close image window
    
    #Remove noise with median param
    K = myImFilter(B, 'median')
    K= K.astype(np.uint8)
    cv2.imshow('Median image',K) #show image
    cv2.waitKey(0) #wait for key press
    cv2.destroyAllWindows() #close image window
    
    #Remove noise with mean param
    C = myImFilter(A, 'mean')
    cv2.imshow('Mean image',C) #show image
    cv2.waitKey(0) #wait for key press
    cv2.destroyAllWindows() #close image window


if __name__ == "__main__":
    main()