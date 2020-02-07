import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    img = np.copy(input_image)
    # TODO: Preprocess the image for the pose estimation model
    # hape: [1x3x256x456] 
    
    B = 1 
    C = 3
    H = 256
    W = 456
    
    img = cv2.resize(img,(W,H))
    img = img.transpose((2,0,1))
    img = img.reshape(B,C,H,W)
        
    return img


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    img = np.copy(input_image)
    
    # TODO: Preprocess the image for the text detection model
    # shape: [1x3x768x1280] 
    
    B = 1 
    C = 3
    H = 768
    W = 1280

    img = cv2.resize(img,(W,H))
    img = img.transpose((2,0,1))
    img = img.reshape(B,C,H,W)
    
    return img


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    img = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    # shape: [1x3x72x72]
        
    B = 1 
    C = 3
    H = 72
    W = 72
    
    img = cv2.resize(img,(W,H))
    img = img.transpose((2,0,1))
    img = img.reshape(B,C,H,W)
    
    return img
