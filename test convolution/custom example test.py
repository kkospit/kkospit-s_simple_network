import simple_network_with_numpy as sn
import TEST_convolution_net_main as cn
from PIL import Image, ImageOps
import numpy as np
import csv
import sys

####################
#скорость обучения#
###################
global_alpha = 0.04

###################
#Полносвязная сеть#
###################
struct = ((4, 0, "relu"),  (1, 0, "sigmoid"))	
net = sn.SimpleNet(struct, alpha = global_alpha, weights_file = None)  
#net = sn.SimpleNet(struct, alpha = global_alpha, weights_file = "math_symbols_weights_conv_classifier.npz")  


#################
#Свёрточная сеть#
#################
																												             
layer1 = (1, 3, 1, 1, 2, "relu") # слоёв на входе, размер ядер фильтра, stride, padding, слоёв на выходе/количество фильтров, активация           
layer2 = (2, 3, 2, 1, 1, "relu") # слоёв на входе, размер ядер фильтра, stride, padding, слоёв на выходе/количество фильтров, активация           

# здесь важно указать функцию активации такую же, как и на первом слое в полносвязной сети

conv_struct = (layer1,layer2)#, layer3, layer4)
conv_net = cn.ConvNet(net, conv_struct, global_alpha, weights_file=None)
#conv_net = cn.ConvNet(net, conv_struct, global_alpha, weights_file=("math_symbols_weights_conv_main.npz"))

a = np.array([[1,2,-1],[2,-3,0.5], [0,0.5, -1.5]])

w1 = np.array([[[[0.1, 0.2, -1], [-0.3, -1, 0.5], [1, 0.5, -1]]], 
				[[[1, -0.2, 0], [-0.9, -0.5, 1], [0.8, 0.5, -0.1]]]])

w2 = np.array([[[[0.2, 0.2, 0.1], [0.1, 1.0, 0], [-1.0, -0.2, -0.1]], 
				[[2.0, 1.0, 1.0], [-0.3, -0.4, 1.0], [0.4, 0.5, 1.0]]]])

				
wfc = np.array(	[[0.2, 0.6, 1.0, -0.7, ]])
print(wfc.shape)
print(w1.shape)
print(w2.shape)

for z in conv_net.weights:
	print(z.shape, "z")
for z in net.weights:
	print(z.shape, "z")
	
net.weights = [wfc]
conv_net.weights = [w1, w2]	

conv_net.net.out_true = [0, 1]
conv_net.layers[0] = a[None, ...]
conv_net.conv_forward()
conv_net.shape_backup = conv_net.layers[-1].shape
#print(conv_net.layers[1])
#'''
sn.error_func = "MSE"
net.out_true = [1]
net.layers[0] = conv_net.layers[-1].reshape(-1)
#print(net.layers)
print(net.forward("test"))
#print(net.layers)

net.backward()
#print(net.deltas)
#print(net.weights)

print("first layer deltas ",net.calc_hidden_deltas(0))

conv_net.conv_backward()
#print("*"*20)
#print(conv_net.deltas)
print("*"*20)
print(conv_net.mods)

conv_net.update_conv_weights()
#print("@"*29)
#print(conv_net.weights)



net.update_weights()

conv_net.weights.clear()

#'''
