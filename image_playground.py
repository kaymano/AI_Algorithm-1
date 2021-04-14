from PIL import Image
import numpy as np
import os

IMG_WIDTH = 200
IMG_HEIGHT = 150
image_path = 'images'

for image_name in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path,f))]:
    x = Image.open(os.path.join(image_path,image_name))
    s = x.size
    new_s = (s[0]-s[0]%IMG_WIDTH,s[1]-s[1]%IMG_HEIGHT)
    x = x.resize(new_s)
    x = np.array(x)
    x_step = IMG_WIDTH
    y_step = IMG_HEIGHT
    a = 0
    images = []
    print(x_step)
    while a<new_s[1]:
        band = x[a:a+y_step]
        b=0
        while b<new_s[0]:
            images.append(band[:,b:b+x_step])
            b+=x_step
            print(a,b)
        a += y_step
    c = 0
    for z in images:
        z = Image.fromarray(z)
        os.makedirs(os.path.join(image_path,'sub',image_name),exist_ok=True)
        z.save(f"{image_path}/sub/{image_name}/grid_{c}.png")
        c+=1


# print(x)