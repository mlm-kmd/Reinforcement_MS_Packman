import random
import gym
import numpy as np
import cv2
import Network
from PIL import Image
from collections import deque

class preprocesser:    
    def process_frame(self, frame, image_size,crop_size):
        image = (image_size[0],image_size[1])
        frame = cv2.resize(frame,dsize=image)
        image = Image.fromarray(frame,'RGB')
        image = image.crop((0,0,image_size[0],image_size[1]-crop_size))
        image = image.convert('L')
        #image.save('my.png')
        #image.show()
        
        img = np.array(image)

        #Normalise
        normalised_image = np.expand_dims(img.reshape((image_size[0],image_size[1]-crop_size,image_size[2])), axis=0) / 255
        return normalised_image