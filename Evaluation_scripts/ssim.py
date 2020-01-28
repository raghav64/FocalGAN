

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from keras.datasets import cifar10

#Specify number of classes and path of generated images upon which to calculate ssim 
number_of_classes_imbalance = 9

data = np.load(str(number_of_classes_imbalance) + '_class_drop_99/generated_imgs.npy')
total_sum = 0
total_count = 0

for idx in range(number_of_classes_imbalance):
    y_test = [idx]*100
    x_test = data[idx * 100 : (idx+1)*100 ]
    ssim_score = 0   
    ct = 0
    for jdx in range(100):
        for kdx in range(100):
            ssim_score+=ssim(x_test[jdx] , x_test[kdx] , multichannel=True)
            ct+=1
    
    total_sum+=ssim_score
    total_count+=ct
    
    print("Ssim for class "+ str(idx) + "-->" + str(ssim_score/ct))
    
print("Avg Score for imbalace class -->" + str(total_sum/total_count))





