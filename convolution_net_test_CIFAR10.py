import numpy as np 
import simple_network_with_numpy as sn
import pickle
# функция свёртки
class ConvNet():
	
	def __init__(self, fc_net, structure: tuple, alpha=0.5,  weights_file:tuple = None):
		self.conv_struct = structure	
		self.conv_alpha = alpha	
		# хранение значений слоёв
		self.layers = [[0] for _ in range(len(conv_struct)+1)]
		self.deltas =  [] # дельты слоёв свёрточной сети
		self.mods = [] # модификаторы к весам свёрточной сети
		self.weights = []
		self.net = fc_net
				
		if weights_file:
			# загружаем веса
			#weights_filename = "weights_conv_main(2).npz"
			#weights_classifier = "weights_conv_classifier(2).npz"
			weights_file_main = np.load(weights_file(0)) # для свёрточной части
			weights_file_classifier = np.load(weights_file(1)) # для полносвязной
			
			for w in weights_file_main:
				self.weights.append(weights_file_main[w])
			
			# заменяем веса в полносвязной сети
			self.net.weights = []
			for w in weights_file_classifier:
				self.net.weights.append(weights_file_classifier[w])
				

		else:
			# генерация весов/фильтров
			for layer in conv_struct:
				w = np.random.uniform(low=-0.5, high=0.5, size=(layer[4], layer[0], layer[1], layer[1]))
				self.weights.append(w)


	def convolve(self, inp, filtr,  size=3, stride=1, padding=0):
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
		output = np.zeros((output_size, output_size), dtype=np.float64)

		# окружаем нулями
		if padding > 0:
			temp = np.zeros((inp.shape[0], inp.shape[1] + padding*2, inp.shape[2] + padding*2), dtype=np.float64)
			temp[:, padding: -padding, padding: -padding] = inp
			inp = temp

		for idx_row	in range(0, inp.shape[1]-size+1, stride):
			for idx_col in range(0, inp.shape[2]-size+1, stride):
				output[int(idx_row/stride), int(idx_col/stride)] += np.sum(inp[:, idx_row: idx_row+size, idx_col: idx_col+size] * filtr )#+ bias)
		
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
			temp = self.convolve(self.layers[layer_idx-1], self.weights[layer_idx-1][0], size = layer[1], stride = layer[2], padding = layer[3])[None, ...]  # для первого фильтра
			for filtr in self.weights[layer_idx-1][1:]: # для остальных
				
				temp = np.vstack((temp, self.convolve(self.layers[layer_idx-1], filtr,  size = layer[1], stride = layer[2], padding = layer[3])[None, ...]))
			self.layers[layer_idx] = self.activation(temp, self.conv_struct[layer_idx-1][5]) # активация
			
			
	
	def conv_backward(self):
		# для последнего слоя свёрточной сети тоже можем посчитать дельты, как для полносвязного
		input_conv_layer_deltas = self.net.calc_hidden_deltas(0) # явно вызывая метод, который обычно вызывается из backward, указываем 0 как индекс входного слоя
		input_conv_layer_deltas = input_conv_layer_deltas.reshape(conv_net.shape_backup)# получили дельты последнего свёрточного слоя
		self.deltas.append(input_conv_layer_deltas)

		# считаем остальные дельты
		for idx in range(len(self.layers)-1, 1, -1):
			self.deltas.insert(0, self.calc_hidden_conv_deltas(idx-1))
		
		# считаем модификаторы весов
		for d_idx, _ in enumerate(self.deltas):
			self.mods.append(self.calc_conv_mods(d_idx))
			
			
	def calc_hidden_conv_deltas(self, idx):
		temp = []

		for d_idx, d in enumerate(self.deltas[0]):
			pad = self.weights[idx][-1].shape[1] - 1 - self.conv_struct[idx][3]
			
			if self.conv_struct[idx][2] > 1 and d.shape[1]!=1: #stride	
				d = self.sparse_deltas(d, idx)

			for w_idx, w in enumerate(self.weights[idx][d_idx]):	
				if len(temp) != 0:
					temp = np.vstack((temp, self.convolve(d[None, ...], np.flip(w, (0,1)),  size = self.conv_struct[idx][1], stride = 1, padding = pad)[None, ...]))
				else:
					temp = self.convolve(d[None, ...], np.flip(w, (0,1)), size = self.conv_struct[idx][1], stride = 1, padding = pad)[None, ...]

		temp.shape = (self.weights[idx].shape[0], self.layers[idx].shape[0], self.layers[idx].shape[1], self.layers[idx].shape[2])
		temp = np.sum(temp, 0)	
		
		return temp*self.derivative(self.layers[idx], self.conv_struct[idx][5])


	def sparse_deltas(self, deltas, idx):
		z = np.zeros((deltas.shape[0]+((deltas.shape[0]-1)*(self.conv_struct[idx][2]-1)), deltas.shape[1]+((deltas.shape[1]-1)*(self.conv_struct[idx][2]-1))))
		z[slice(0, len(z[0]), self.conv_struct[idx][2]), slice(0, len(z[0]), self.conv_struct[idx][2])] = deltas
		return z


	def calc_conv_mods(self, idx):
		temp = []
		# print(idx, layers[idx].shape, deltas[idx].shape, weights[idx].shape)
		for d_idx, d in enumerate(self.deltas[idx]):
			pad = self.conv_struct[idx][3]
		
			if conv_struct[idx][2] > 1: #stride
				d = self.sparse_deltas(d, idx)

			for l_idx, l in enumerate(self.layers[idx]):
		
				if len(temp) != 0:
					temp = np.vstack((temp, self.convolve(l[None, ...], d,  size = d.shape[-1], stride = 1, padding = pad)[None, ...]))
				else:
					temp = self.convolve(l[None, ...], d,  size = d.shape[-1], stride = 1, padding = pad)[None, ...]
		
		temp.shape = self.weights[idx].shape
		return temp

		
	def update_conv_weights(self):
		for w_idx, _ in enumerate(self.weights):
			self.weights[w_idx] -= self.mods[w_idx] * self.conv_alpha


####################
#скорость обучения#
###################
global_alpha = 0.03

###################
#Полносвязная сеть#
###################
struct = ((12, 0, "relu"), (10, 0, "tanh"), (10, 0, "softmax"))	
net = sn.SimpleNet(struct, alpha = global_alpha, weights_file=None)  


#################
#Свёрточная сеть#
#################
																												             
layer1 = (3, 7, 1, 1, 4, "relu") # слоёв на входе, размер ядер фильтра, stride, padding, слоёв на выходе/количество фильтров, активация           
layer2 = (4, 2, 2, 0, 8, "relu")         
layer3 = (8, 3, 1, 1, 10, "relu") # здесь важно указать функцию активации такую же, как и на первом слое в полносвязной сети        
layer4 = (10, 2, 2, 0, 12, "relu") # здесь важно указать функцию активации такую же, как и на первом слое в полносвязной сети        
layer5 = (12, 7, 1, 0, 12, "relu") # здесь важно указать функцию активации такую же, как и на первом слое в полносвязной сети        
conv_struct = (layer1, layer2, layer3, layer4, layer5)
conv_net = ConvNet(net, conv_struct, global_alpha)


############################
#Выбор режима обучение/тест#
############################

mode = "train"
#mode = "test"


if __name__ == "__main__":
	e = [0] # список с ошибками		

	def create_true_vector_cifar(value):
		output = [0.0 for _ in range(10)]
		output[value] = 1.00
		return output

	###################
	##Данные CIFAR10###
	###################
	
	def unpickle_cifar(f):
		with open(f, 'rb') as fo:
			data = pickle.load(fo, encoding='bytes')
		return data

	if mode == "train":
		data = unpickle_cifar("/home/krep_kospit/Desktop/cifar10/cifar-10-batches-py/data_batch_1")
	elif mode == "test":
		data = unpickle_cifar("/home/krep_kospit/Desktop/cifar10/cifar-10-batches-py/test")


	for idx, _ in enumerate(data[b"labels"][:10]):
		print(data[b"labels"][idx])
		print(data[b"filenames"][idx])

	labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
			  5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

	len_data = len(data[b"labels"]) # 10000 в одном файле
	
	
	for i in range(10):
		print("*"*10, i, "*"*10)
		for idx, label in enumerate(data[b"labels"][:2]):
			
			out_true = create_true_vector_cifar(int(label))
			# print(label, out_true, labels[label])
			net.out_true = out_true
			
			inp = data[b"data"][idx].reshape(-1, 32, 32)

			conv_net.layers[0] = inp.squeeze()
			conv_net.conv_forward()
			
			#print(layers[-1].shape)	
			conv_net.shape_backup = conv_net.layers[-1].shape
			net.layers[0] = conv_net.layers[-1].squeeze()
			# считаем выход полносвязного слоя и ошибку
			errors = net.forward()
			
			e.append((errors))

			if mode == "train":
				# считаем дельты
				net.backward()
				conv_net.conv_backward()
				net.update_weights()
				conv_net.update_conv_weights()
					
				net.deltas.clear()
				conv_net.deltas.clear()
				conv_net.mods.clear()
			#print(errors)
			if i % 2 == 0 and i !=0:
				pred_idx = net.layers[-1].tolist().index(max(net.layers[-1]))
				print("epoch:", i, f"image: {idx}/{len_data}", "errors:", errors, "prediction:", f"{pred_idx}/{labels[pred_idx]}", "true", f"{label}/{labels[label]}")

	if mode == "train":
		np.savez("cifar10_weights_conv_main", *conv_net.weights)	
		np.savez("cifar10_weights_conv_classifier", *net.weights)	

	print("mid loss:", sum(e)/len(e))



