import os
from PIL import Image
import glob

size = (200,200)
path = 'C:/Users/duboisrouvray/Documents/Entrainement_ML/WondersofWorld/'
for folder in os.listdir(path) :
    for i,file in enumerate(glob.iglob(os.path.join(path+folder, "*.jpg"))):
        img = Image.open(r'{}'.format(file))
        img = img.resize(size).convert('RGB')
        img.save('{}'.format(file))