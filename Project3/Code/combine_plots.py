"""
check
"""
import sys
from PIL import Image

method=3
if method==1: #SVM
    image_list=["../Results/Confusion/conf_SVM_nores.jpg","../Results/Confusion/conf_SVM_adasyn.jpg"]
if method==2: #NN
    image_list=["../Results/Confusion/conf_NN_nores.jpg","../Results/Confusion/conf_NN_adasyn.jpg"]

if method==3: #grid results NN
    image_list=["../Results/Confusion/conf_NN_grid_nores.jpg","../Results/Confusion/conf_NN_grid_adasyn.jpg"]

images = [Image.open(x) for x in image_list]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

if method==1: #SVM
    new_im.save('../Results/confusions_SVM.pdf')
if method==2: #NN
    new_im.save('../Results/confusions_NN.pdf')
if method==3: #NN
    new_im.save('../Results/confusions_NN_grid.pdf')
