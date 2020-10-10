from pathlib import Path
from PIL import Image, ImageOps
import random as rnd
import numpy as np

size = 37
'''
img = Image.open("alpha.jpg").convert("L")
box = ImageOps.invert(img).getbbox()
#img = img.resize((size, size))
'''

def sqim(img, dx, dy):
	
	img = img.resize((size-int(size/2)*dx, size-int(size/2)*dy))
	array = np.array(img) 
	#temp = np.zeros((size, size)) + 255
	temp = np.full((size, size), 255, dtype="uint8")
	#print(array.shape, dy, dx)
	temp[0+int(array.shape[0]/2)*dy : int(array.shape[0]/2)*dy+array.shape[0], 0+int(array.shape[1]/2)*dx : int(array.shape[1]/2)*dx+array.shape[1]] = array

	img = Image.fromarray(temp)
	#img.show()
	return img


def squeeze_image(*img):
	temp = []
	for i in img:
		chs = rnd.randint(0, 1)
		if chs == 0:
			i = sqim(i, 1, 0)
		else:
			i = sqim(i, 0, 1)
				
		temp.append(i.resize((size, size)))
	
	return temp
	
	
path = Path("./old_data")

for p in path.iterdir():
	new = Path("./new_data/" + p.name)
	new.mkdir(parents=True, exist_ok=True)

	for image in p.glob("*.png"):	
		
		img = Image.open(image)
	
		# https://stackoverflow.com/questions/9506841/using-python-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image/50090612#50090612
		fn = lambda x : 0 if x >= 155 else 255
		
		img = img.convert('L')#.point(fn)

		box_img_crop = img.convert('L').point(fn)

		#box = ImageOps.invert(img).getbbox()
		box = box_img_crop.getbbox()
				
		
		
		direct_full = rnd.choice((-1, 1))
		#direct_crop = rnd.choice((-1, 1))
		img_full_rot = img.rotate(rnd.randint(5, 15)*direct_full, fillcolor=(255), expand=False)
		#img_crop_rot = img_crop.rotate(rnd.randint(5, 15)*direct_crop, fillcolor=(255), expand=False)
		
		#img_full_sq, img_crop_sq = squeeze_image(img_full, img_crop)
		img_full_sq = squeeze_image(img)[0]#, img_crop)
		
		#img_flip_full = img_full.transpose(Image.FLIP_TOP_BOTTOM)
		#img_flip_crop = img_full.transpose(Image.FLIP_TOP_BOTTOM)
		
		img_full = img.resize((size, size))#,  box = box)
		img_crop = img.resize((size, size),  box = box)
		
		img_full_rot = img_full_rot.resize((size, size))
		img_full_sq = img_full_sq.resize((size, size))
		
		'''
		img_full = img_full.convert('L').point(fn)
		img_crop = img_crop.convert('L').point(fn)
		img_full_rot = img_full_rot.convert('L').point(fn)
		#img_crop_rot = img_crop_rot.convert('L').point(fn)
		img_full_sq = img_full_sq.convert('L').point(fn)
		#img_crop_sq = img_crop_sq.convert('L').point(fn)
		'''
		img_full = ImageOps.invert(img_full)
		img_crop = ImageOps.invert(img_crop)
		img_full_rot = ImageOps.invert(img_full_rot)
		img_full_sq = ImageOps.invert(img_full_sq)
		
		
		#img_2 = img.rotate(+15)
		print(img.format, img.size, img.mode, list(image.parents))
		name = image.name.split(",")[0]
		#print(image.name.split("."))
		#input("asd")
		#img.save(f"new_data/{p.name}/{image.name}")
		img_full.save(f"new_data/{p.name}/{image.name.split('.')[0]}_full.{image.name.split('.')[1]}")
		img_crop.save(f"new_data/{p.name}/{image.name.split('.')[0]}_crop.{image.name.split('.')[1]}")
		
		img_full_rot.save(f"new_data/{p.name}/{image.name.split('.')[0]}_full_rot.{image.name.split('.')[1]}")
		#img_crop_rot.save(f"new_data/{p.name}/{image.name.split('.')[0]}_crop_rot.{image.name.split('.')[1]}")
		
		img_full_sq.save(f"new_data/{p.name}/{image.name.split('.')[0]}_full_sq.{image.name.split('.')[1]}")
		#img_crop_sq.save(f"new_data/{p.name}/{image.name.split('.')[0]}_crop_sq.{image.name.split('.')[1]}")
		#img_flip_full.save(f"new_data/{p.name}/{image.name.split('.')[0]}_flip_full.{image.name.split('.')[1]}")
		#img_flip_crop.save(f"new_data/{p.name}/{image.name.split('.')[0]}_flip_crop.{image.name.split('.')[1]}")
		
		#img_2.save(f"new_data/{p.name}/{name}_2.png")


#'''
		
'''

path = Path("./new_data")

for p in path.iterdir():
	for image in p.glob("*.png"):
		img = Image.open(image)
		print(img.format, img.size, img.mode, list(image.parents))
'''
