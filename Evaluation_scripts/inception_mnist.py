
import numpy as np
from keras.models import model_from_json
from keras.datasets import mnist

def calculate_inception_score(p_yx , eps = 1e-16):
    #Calculate p(y)
    p_y = np.expand_dims(np.mean(p_yx , axis = 0) , 0)
    
    #Kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    
    # undo the logs
    is_score = np.exp(avg_kl_d)
    
    return is_score

# load json and create model
Path_to_model = ''
Path_to_model_weights = ''
Path_to_generated_images = ''

json_file = open(Path_to_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(Path_to_model_weight)

data = np.load(Path_to_generated_images)
data = data/255.
p = loaded_model.predict_proba(data)
inc = calculate_inception_score(p)
print(inc)

