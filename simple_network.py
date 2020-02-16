from random import random, choice, shuffle
import json
import math


class SimpleNet():
	
	def __init__(self, structure: tuple, alpha=0.5, weights_file = False):
		
		assert (type(alpha) == int or type(alpha) == float), "коэффициент обучения должен быть числом!"
		self.alpha = alpha
	
		self.errors = []
		self.deltas = [] # дельты слоёв
					
		#assert (len(inp) == structure[0][0]), "Число входных параметров должно совпадать с числом нейронов первого слоя!"
		
		self.structure = structure
		self.num_layers = len(structure)
		self.layers =  [[0] for _ in range(self.num_layers)] # не обязательно вручную задавать структуру, всё равно слои обновятся как надо из-за структуры списка весов
		self.neurons_at_layer = tuple([num[0] for num in self.structure])
		self.num_hidden_layers = self.num_layers-2 # минус вход и выход 
		
		# получим смещения
		self.bias = self.get_bias()
		
		# загрузим или сгенерируем веса
		if weights_file:
			with open(weights_file, "r") as read_file: 
				self.weights = json.load(read_file)
		
		else:
			
			self.weights = []				
			for i in range(self.num_layers-1):
				self.weights.append(([[random()/100]*self.neurons_at_layer[i] for _ in range(self.neurons_at_layer[i+1])])) # заполнять рандомными весами		
			
			
							
		# вектор правильных ответов
		self.out_true = ()
		

	#######################################
	######функции для прямого прохода######
	#######################################
	# получаем смещения
	def get_bias(self):
		bias = []
		for pair in self.structure:
			if len(pair) == 1:
				bias.append(0)
			else:
				bias.append(pair[1])
		
		return bias
		

	# считаем вход на нейроны следующего слоя
	def calc_input(self, inp, weights, bias=0):
		output = [0]*len(weights)
		for idx, weight in enumerate(weights):
			output[idx] = sum([inp[i]*w  for i, w in enumerate(weight)]) + bias
		return output
		
		
	# функция активации	
	def sigmoid(self, x):
		return 1/(1+math.exp(-x))
	
	def tanh(self, x):
		return (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
	

	# считаем выход нейронов слоя	
	def calc_output(self, enter, func="sigmoid"):

		output = [0]*len(enter)

		if func == "sigmoid":
			for idx, i in enumerate(enter):
				output[idx] = self.sigmoid(i)
			return output
		elif func == "tanh":
			for idx, i in enumerate(enter):
				output[idx] = self.tanh(i)
			return output

	# вектор ошибок
	def calc_errors(self, output):
		# MSE 
		errors = sum([(self.out_true[idx] - o)**2 for idx, o in enumerate(output)])/len(output)
		return errors


	# прямой проход
	def forward(self):
		assert (len(self.layers[0]) == self.structure[0][0]), "количество входных параметров должно совпадать с количеством входных нейронов!"
		
		# скрытые слои
		for idx, layer in enumerate(self.layers[1:], start=1):
			cur_inp = self.calc_input(self.layers[idx-1], self.weights[idx-1], self.bias[idx-1])
			cur_out = self.calc_output(cur_inp, self.structure[idx][2])	
			self.layers[idx] = cur_out

		errors = self.calc_errors(self.layers[-1])
		return errors
		
	
	#######################################
	#функции для обратного распространения#
	#######################################
	# производная 
	def derivative(self, x, func="sigmoid"):
		if func == "sigmoid":
			return x*(1-x)
		elif func == "tanh":
			return 1 - x**2
	
	def transpose(self, matrix):
		output = []
		
		for idx, _ in enumerate(matrix[0]):
			output.append([e[idx] for e in matrix])
		
		return output

	# матричное умножение
	def matrix_mul(self, a, b):
		output = [[]*len(a)]
		b = self.transpose(b)
		for idx_row_a, row_a in enumerate(a):
			for idx_row_b, row_b in enumerate(b):
				output[idx_row_a].append(sum([elem_a*row_b[idx] for idx, elem_a in enumerate(row_a)]))	
		
		return output
		

	# дельты выходного слоя, так как там производная по функции ошибки, а не активации
	def calc_output_deltas(self):
		out_deltas = []
		for idx, true in enumerate(self.out_true):
			out_deltas.append((self.layers[-1][idx] - true)*self.derivative(self.layers[-1][idx], self.structure[-1][2]))
		return out_deltas


	# отдельно считаем дельты для скрытого слоя
	def calc_hidden_deltas(self, layer_idx):
		# сперва посчитаем производные функций активаций от входных значений нейронов этого слоя
		derivatives_current_layer = []
		
		for idx, neuron in enumerate(self.layers[layer_idx]):
			derivatives_current_layer.append(self.derivative(neuron, self.structure[layer_idx][2]))
		
		# умножаем дельты следующего слоя на веса от данного слоя к следующему
		deltas_prev_mul_curr_weights = self.matrix_mul([self.deltas[-1]], self.weights[layer_idx]) # dpcw для краткости
		
		# соответствующие производные умножаем на соответствующие суммы дельт следующего слоя
		curr_deltas = [dpcw*derivatives_current_layer[neuron_idx] \
					   for neuron_idx, dpcw in enumerate(*deltas_prev_mul_curr_weights)]

		return curr_deltas


	# то, чем будем корректировать веса
	def calc_modificators(self, deltas, layer_idx):
		output = []
		# выход нейрона пред. слоя*дельту текущего*коэфф. скорости обучения
		output.extend([[l*d*self.alpha for l in self.layers[layer_idx]] for d in deltas])

		return output
		

	# выдача скорректированных весов
	def calc_modified_weights(self, deltas, layer_idx):
		mods = self.calc_modificators(deltas, layer_idx)
		output = []
		# надо функцию для суммы матриц
		for idx, neuron_weights in enumerate(self.weights[layer_idx]):
			output.append(tuple([weight - mods[idx][inner_idx] for inner_idx, weight in enumerate(neuron_weights)]))
		return output
		
		
	# одновление списка весов, когда посчитаны дельты слоёв
	def update_weights(self):
		for idx, d in enumerate(reversed(self.deltas)):# т.к. сперва в список добавлялись дельты последнего слоя
			new_weights = self.calc_modified_weights(d, idx)
			self.weights[idx] = new_weights


	# обратное распространение ошибки
	# обычный градиентный спуск
	def backward(self):
		# находим дельты выходного слоя	
		output_deltas = self.calc_output_deltas()
		self.deltas.append(output_deltas)

		# находим дельты скрытого слоя
		for layer_idx in range(self.num_hidden_layers, 0, -1):	
			hidden_deltas = self.calc_hidden_deltas(layer_idx)
			self.deltas.append(hidden_deltas)
		

## test
'''
# данные из статьи Matt Mazur про backpropagation
struct = ((2, 0.35, "tanh"), (2, 0.6, "tanh"), (2,0, "tanh")) # кол-во нейронов, величина смещения
s = SimpleNet(struct)

# временно для тестов вставить эти веса
s.weights = [	 #1    #2    #3->h11  #1    #2   #3->h12
				   ((0.15, 0.20), (0.25, 0.30)), 
				   ((0.40, 0.45), (0.50, 0.55))]
s.layers[0] = (0.05, 0.1)
s.out_true = (0.01,  0.99)
s.forward()
s.backward()
s.update_weights()
s.deltas.clear()
print(s.weights)
'''

# должны получится веса в тесте
#[[(0.1497807161327628, 0.19956143226552567), (0.24975114363236958, 0.29950228726473915)], [(0.35891647971788465, 0.4086661860762334), (0.5113012702387375, 0.5613701211079891)]]




# тренировка на 20к картинках MNIST
data = open("mnist_train_small.csv", "r").readlines()
len_data = len(data)
			#N  #bias #f.activ.
struct = ((784, 0, "sigmoid"), (50,0,"sigmoid"), (20,0,"sigmoid"), (10,0,"sigmoid"))	
net = SimpleNet(struct)

n_iter = 0 # ставим 0, если сразу хотим тестить

def create_true_vector_mnist(value):
	output = [0.0 for _ in range(10)]
	output[value] = 1.00
	return output


for iter_idx in range(n_iter):
	shuffle(data)
	for idx, row in enumerate(data):
		out_true = create_true_vector_mnist(int(row[0]))
		net.out_true = out_true
		row = row.split(",")
		inp = [int(s)/255 for s in row[1:]] 
		net.layers[0] = inp
		
		errors = net.forward()
		net.backward()
		# обновляем веса	
		net.update_weights()
		# очищаем дельты, чтобы потом по-новой записать вычисленные с учётом обновлённых весов
		net.deltas.clear()
	
		if idx%500==0:
			row = choice(data)
			out_true = create_true_vector_mnist(int(row[0]))
			net.out_true = out_true
			row = row.split(",")
			inp = [int(s)/255 for s in row[1:]] 	
			net.layers[0] = inp
			out = net.forward()
			print("epoch:", iter_idx, "current_index/data", f"{idx}/{len_data}", "true:",out_true.index(max(out_true)), "prediction:", net.layers[-1].index(max(net.layers[-1])) )
			print("error", errors, net.layers[-1])
			


# тестирование
print("$"*20, "TEST", "$"*20)
# можем загрузить веса. обучал на google colab
read_file = open("weights_file.json", "r")
weights = json.load(read_file)
net.weights = weights 

counter = {0:{"total":0, "true":0},
			1:{"total":0, "true":0},
			2:{"total":0, "true":0},
			3:{"total":0, "true":0},
			4:{"total":0, "true":0},
			5:{"total":0, "true":0},
			6:{"total":0, "true":0},
			7:{"total":0, "true":0},
			8:{"total":0, "true":0},
			9:{"total":0, "true":0},
}


def show_number(inp):
	for idx, pixel in enumerate(inp):
		if idx%28 == 0:
			print()
		if pixel >0:
			pixel = 1
		else:
			pixel = 0
			
		print(pixel, end="")
	print()


data = open("mnist_train_100.csv", "r").readlines()[:10]
for idx, row in enumerate(data):
	out_true = create_true_vector_mnist(int(row[0]))
	net.out_true = out_true
	row = row.split(",")
	inp = [int(s)/255 for s in row[1:]] 
	net.layers[0] = inp
	
	out = net.forward()
	show_number(inp)
	
	print("true:", int(row[0]), "predict:", net.layers[-1].index(max(net.layers[-1])) )
	counter[int(row[0])]["total"] += 1
	if int(row[0]) == net.layers[-1].index(max(net.layers[-1])):
		counter[int(row[0])]["true"] += 1

for key in counter:		
	print(key, ":", counter[key])




# test на картинках
print()
print("$"*20, "CUSTOM IMAGE TEST", "$"*20)

import numpy as np

from PIL import Image, ImageOps
img = Image.open("test.png").convert("L") # L == black and white
#img = ImageOps.invert(img) # раскомментировать, если фон - белый, цифра - чёрная
#img = ImageMath.abs()
img = img.resize((28,28), Image.BILINEAR)
img = np.asarray(img).ravel()/255

'''
# чтобы сохранить картинку из mnist 
row = choice(data)
row = row.split(",")
inp = [float(s) for s in row[1:]] 
inp = np.array(inp).reshape(28,28).astype(np.uint8)

i = Image.fromarray(inp)
i.save("222.png")
# работает же
'''

show_number(img)	
	
net.layers[0] = img
out = net.forward()
print("predict:", net.layers[-1].index(max(net.layers[-1])) )
print(net.layers[-1])
