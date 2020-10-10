import numpy as np 
import simple_network_with_numpy as sn
import pickle
import time


## попытка внести изменения в градиентный спуск - в update


rng = np.random.RandomState(29)
lwi = 0.5 # limit_weights_initializations

class ConvNet():
	
	def __init__(self, fc_net, structure: tuple, alpha,  weights_file:str = None):
		self.conv_struct = structure	
		self.conv_alpha = alpha	
		# хранение значений слоёв
		self.layers = [[0] for _ in range(len(self.conv_struct)+1)]
		self.deltas =  [] # дельты слоёв свёрточной сети
		self.mods = [] # модификаторы к весам свёрточной сети
		self.weights = []
		self.weights_store = []
		self.net = fc_net
				
		if weights_file:
			# загружаем веса
			weights_file_main = np.load(weights_file) # для свёрточной части
			#weights_file_main = np.load(weights_file[0]) # для свёрточной части
			#weights_file_classifier = np.load(weights_file[1]) # для полносвязной
			
			for w in weights_file_main:
				self.weights.append(weights_file_main[w])
			
			# заменяем веса в полносвязной сети
			#self.net.weights = []
			#for w in weights_file_classifier:
			#	self.net.weights.append(weights_file_classifier[w])
				
		else:
			# генерация весов/фильтров
			for layer in self.conv_struct:
				w = rng.uniform(low=-lwi, high=lwi, size=(layer[4], layer[0], layer[1], layer[1]))
				self.weights.append(w)
				
		# https://habr.com/ru/post/318970/ начало объяснения алгоритма Нестерова
		for layer in self.conv_struct:
			self.weights_store.append(np.zeros((layer[4], layer[0], layer[1], layer[1])))


	def convolve(self, inp, filtr,  size=3, stride=1, padding=0, mode="forward"):
		# считаем размер следующего слоя
		output_size = self.calc_output_size(inp.shape[1], size, stride, padding)	
		#'''
		if output_size.is_integer():
			output_size = int(output_size)
		else:
			print("Ядро свёртки не охватывает полностью изображение!")
			print("\t", inp.shape)
			raise ValueError
		#'''
		if mode == "forward":
			output = np.zeros((filtr.shape[0], output_size, output_size), dtype=np.float64)
		elif mode == "deltas":
			#print(type(filtr), type(inp))
			output = np.zeros((filtr.shape[0]*inp.shape[0], output_size, output_size), dtype=np.float64)
			#print(filtr.shape, inp.shape,"FS IS")
			filtr = filtr.reshape((filtr.shape[0], 1, filtr.shape[1], filtr.shape[2]))
			#print(filtr.shape, "FS")
		
		
		# окружаем нулями
		if padding > 0:
			#if mode == "deltas": print(inp.shape,"%%%%%%%%%", inp)
			temp = np.zeros((inp.shape[0], inp.shape[1] + padding*2, inp.shape[2] + padding*2), dtype=np.float64)
			temp[:, padding: -padding, padding: -padding] = inp
			inp = temp
		
		axis = (1,2,3)
		#if mode == "deltas":
		#	axis = axis[:-1]	
		
		for idx_row	in range(0, inp.shape[1]-size+1, stride):
			for idx_col in range(0, inp.shape[2]-size+1, stride):
				if mode == "forward":
					output[:, int(idx_row/stride), int(idx_col/stride)] += np.sum(inp[:, idx_row: idx_row+size, idx_col: idx_col+size] * filtr, axis=axis )#[None,...]#+ bias)
				elif mode == "deltas":
					#print(np.sum(inp[:, idx_row: idx_row+size, idx_col: idx_col+size] * filtr, axis = (2,3)).shape, "@@@@@@@@@@@@")
					#print(inp[:, idx_row: idx_row+size, idx_col: idx_col+size].shape, filtr.shape, "!!!!!!!!!!!!!!")
					output[:, int(idx_row/stride), int(idx_col/stride)] += np.sum(inp[:, idx_row: idx_row+size, idx_col: idx_col+size] * filtr, axis=(2,3)).squeeze().reshape(-1)

		return output
			
				
	def calc_output_size(self, shape, size, stride, padding):
		return  ((shape - size + 2*padding)/stride) + 1
		

	def activation(self, x, name):
		if name == "sigmoid":
			return 1/(1 + np.exp(-x))
		elif name == "relu":
			return np.maximum(0, x)	
		elif name == "tanh":
			#return (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
			return np.tanh(x)
			
			
	def derivative(self, x, name):
		if name == "sigmoid":
			return x*(1-x)
		elif name == "relu":	
			return (x > 0).astype(int)
		elif name == "tanh":
			return 1 - x**2


	def conv_forward(self):
		# forward pass 
		for layer_idx, layer in enumerate(self.conv_struct, start = 1):
			temp = self.convolve(self.layers[layer_idx-1], self.weights[layer_idx-1], size = layer[1], stride = layer[2], padding = layer[3])#[None, ...]  # для первого фильтра
			#print(temp)
			self.layers[layer_idx] = self.activation(temp, self.conv_struct[layer_idx-1][5]) # активация
			#print(temp, "temp")
			#print(self.layers[layer_idx])
			#input("asdasd")

	
	def conv_backward(self):
		# для последнего слоя свёрточной сети тоже можем посчитать дельты, как для полносвязного
		input_conv_layer_deltas = self.net.calc_hidden_deltas(0) # явно вызывая метод, который обычно вызывается из backward, указываем 0 как индекс входного слоя
		#print(input_conv_layer_deltas, "deltas last conv layer")
		input_conv_layer_deltas = input_conv_layer_deltas.reshape(self.shape_backup)# получили дельты последнего свёрточного слоя
		self.deltas.append(input_conv_layer_deltas)

		# считаем остальные дельты
		for idx in range(len(self.layers)-1, 1, -1):
			self.deltas.insert(0, self.calc_hidden_conv_deltas(idx-1))
		# считаем модификаторы весов
		for d_idx, _ in enumerate(self.deltas):
			self.mods.append(self.calc_conv_mods(d_idx))
				
					
	def calc_hidden_conv_deltas(self, idx):
		temp = []
		pad = self.weights[idx][-1].shape[1] - 1 - self.conv_struct[idx][3]
		
		#for d_idx, d in enumerate(self.deltas[0]):
		#  разрежаем карты для обратной свёртки, если они были получены с шагом больше 1
		#print(d_idx, "IDX D")
		#print(self.deltas[0],self.deltas[0].shape )
		#print(self.weights[idx], self.weights[idx].shape)
		
		filtr = np.flip(self.weights[idx], (2,3)) - np.flip(self.weights_store[idx], (2,3)) * 0.9
		
		if self.conv_struct[idx][2] > 1:
			temp_deltas = self.mod_sparse_deltas(self.deltas[0], idx)
		else:
			temp_deltas = self.deltas[0]
		
		
		#temp_deltas = np.stack((temp_deltas, temp_deltas), axis = 1)
		#temp_deltas = np.moveaxis(temp_deltas, 1, 0)
		temp = self.convolve(inp = temp_deltas, 
								 filtr = filtr.transpose((1, 0, 2, 3)), # транспонирую, чтобы дельта с индексом 0 умножалась на группу весов с тем же индексом. см. старый вариант
								 size = self.conv_struct[idx][1], 
								 stride = 1, 
								 padding = pad, 
								 mode = "forward")
		#input()
		'''
			if self.conv_struct[idx][2] > 1 and d.shape[1]!=1: #stride	
				d = self.sparse_deltas(d, idx)
				#print(d, "SPARSE")
		'''
		'''
			 # пробуем сделать "заглядывание вперёд" по алгоритму Нестерова. надеюсь, я всё правильно понял.
			if len(temp) != 0:
				temp = np.vstack((temp, self.convolve(inp = d[None, ...],
														filtr = np.flip(self.weights[idx][d_idx], (1,2)) - np.flip(self.weights_store[idx][d_idx], (1,2)) * 0.9,
														size = self.conv_struct[idx][1], 
														stride = 1, 
														padding = pad, 
														mode ="deltas")))
			else:
				temp = self.convolve(inp = d[None, ...], 
									 filtr = np.flip(self.weights[idx][d_idx], (1,2)) - np.flip(self.weights_store[idx][d_idx], (1,2)) * 0.9,  
									 size = self.conv_struct[idx][1], 
									 stride = 1, 
									 padding = pad, 
									 mode = "deltas")
		'''
		#print("TEMP",temp.shape)
		#print(self.deltas[0].shape)
		#print(self.weights[idx].shape)
		#print((self.weights[idx].shape[0], self.layers[idx].shape[0], self.layers[idx].shape[1], self.layers[idx].shape[2]))
		#input()
		#temp.shape = (self.weights[idx].shape[0], self.layers[idx].shape[0], self.layers[idx].shape[1], self.layers[idx].shape[2])
		
		#temp = np.sum(temp, 0)	
		#print("SDASDASDASDASDasd")
		#print(temp)
		#input()
		return temp*self.derivative(self.layers[idx], self.conv_struct[idx][5])


	def sparse_deltas(self, deltas, idx):
		z = np.zeros((deltas.shape[0]+((deltas.shape[0]-1)*(self.conv_struct[idx][2]-1)), deltas.shape[1]+((deltas.shape[1]-1)*(self.conv_struct[idx][2]-1))))
		z[slice(0, len(z[0]), self.conv_struct[idx][2]), slice(0, len(z[0]), self.conv_struct[idx][2])] = deltas
		return z
		
	def mod_sparse_deltas(self, deltas, idx):
		#print("IN SPARSE")
		z = np.zeros((deltas.shape[0], deltas.shape[1]+((deltas.shape[1]-1)*(self.conv_struct[idx][2]-1)), deltas.shape[2]+((deltas.shape[2]-1)*(self.conv_struct[idx][2]-1))))
		z[:, slice(0, len(z[0]), self.conv_struct[idx][2]), slice(0, len(z[0]), self.conv_struct[idx][2])] = deltas
		return z

		
	def calc_conv_mods(self, idx):
		temp = []
		#print("DELTAS IN MODS")
		#print(idx,"\n", self.deltas[idx], "\n", self.layers[idx])
		pad = self.conv_struct[idx][3]
	
		if self.conv_struct[idx][2] > 1:
			temp_deltas = self.mod_sparse_deltas(self.deltas[idx], idx)
		else:
			temp_deltas = self.deltas[idx]
		
		#print(pad,"WWWWWWWWWWWWWWW")
		temp = self.convolve(self.layers[idx], temp_deltas,  size = temp_deltas.shape[-1], stride = 1, padding = pad, mode="deltas")[None, ...]
	
		#print("TEMP",temp,  temp.shape)			
		temp = temp.squeeze()
		temp.shape = self.weights[idx].shape
		#print( "MODS", "\n", temp,"\n", temp.shape, self.weights[idx].shape)

		return temp

		
	def update_conv_weights(self):
		for w_idx, _ in enumerate(self.weights):
			
			# добавляем слагаемое 0.9*на модификаторы с прошлого прохода
			v_next =  self.weights_store[w_idx] * 0.9 + self.mods[w_idx] * self.conv_alpha
			self.weights_store[w_idx] = v_next
			
			self.weights[w_idx] -= v_next
			#self.weights[w_idx] -= self.mods[w_idx] * self.conv_alpha

