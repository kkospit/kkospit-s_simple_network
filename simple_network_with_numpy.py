from random import random, choice, shuffle
import json
import numpy as np

# np.show_config()
###!!! не забывать указывать функцию ошибки: MSE или CE!!!

## попытка внести изменения в градиентный спуск - в update

rng = np.random.RandomState(11)

error_func = "CE"
lwi = 0.5 # limit_weights_initializations

class SimpleNet():
	"""Класс, содержащий функции для построения и обучения полносвязной нейронной сети"""
	
	def __init__(self, structure: tuple, alpha=0.5, weights_file = None):
		"""Инициализация нейросети, загрузка или генерация исходных весов
		
		:param structure: описание слоёв сети
		:type structure: tuple
		
		:param alpha: скорость обучения, default to 0.5
		:type alpha: float
		
		:param weights: путь к загружаемым весам, default to None
		:type weights: str 
		"""
		
		assert (type(alpha) == int or type(alpha) == float), "коэффициент обучения должен быть числом!"
		self.alpha = alpha
	
		self.errors = []
		self.deltas = [] # дельты слоёв
		
		self.structure = structure
		self.num_layers = len(structure)
		self.layers =  [[0] for _ in range(self.num_layers)] # не обязательно вручную задавать структуру, всё равно слои обновятся как надо из-за структуры списка весов
		self.neurons_at_layer = tuple([num[0] for num in self.structure])
		self.num_hidden_layers = self.num_layers-2 # минус вход и выход 
		
		#self.last_layers_deltas_store = np.zeros((1, self.structure[-1][0])) # храниение дельт последнего слоя для пакетного градиентного спуска, размерность 1, число нейронов последнего слоя

		# получим смещения
		self.bias = self.get_bias()
		
		self.weights = []#np.array([])
		self.weights_store = []	
		
		for i in range(self.num_layers-1):
			self.weights_store.append(np.zeros((self.neurons_at_layer[i+1], self.neurons_at_layer[i]))) # для модификации градиентного спуска	
			
		# загрузим или сгенерируем веса
		if weights_file:
			weights = np.load(weights_file)
			for w in weights:
				self.weights.append(weights[w])
		
		else:	
			for i in range(self.num_layers-1):
				self.weights.append(rng.uniform(low=-lwi, high=lwi, size=(self.neurons_at_layer[i+1], self.neurons_at_layer[i]))) # заполнять рандомными весами		
		# вектор правильных ответов
		self.out_true = ()
		

	#######################################
	######функции для прямого прохода######
	#######################################
	# получаем смещения # не используется
	def get_bias(self): 
		bias = []
		for pair in self.structure:
			if len(pair) == 1:
				bias.append(0)
			else:
				bias.append(pair[1])
		
		return bias
		
		
	def sigmoid(self, x):
		"""функция активации сигмоида
		:param x: слой в сети
		:type x: numpy array
		
		:rtype: numpy array
		:return: активированные значения нейронов слоя
		"""
		
		return 1/(1 + np.exp(-x))
	
	def tanh(self, x):
		"""функция активации гиперболический тангенс
		:param x: слой в сети
		:type x: numpy array
		
		:rtype: numpy array
		:return: активированные значения нейронов слоя
		"""
		
		return np.tanh(x)

	
	def relu(x):
		"""функция активации ReLU
		:param x: слой в сети
		:type x: numpy array
		
		:rtype: numpy array
		:return: активированные значения нейронов слоя
		"""
		
		return max(0, x)
			

	def softmax(self, array):
		"""функция активации Softmax
		:param x: слой в сети
		:type x: numpy array
		
		:rtype: numpy array
		:return: активированные значения нейронов слоя
		"""
		
		exps = np.exp(array - array.max()) # советуют вычислять так
		return exps / exps.sum()
	
	
	def calc_output(self, array, func_name="sigmoid"):
		"""Активация нейронов нейросети
		
		:param array: слой нейросети
		:type array: numpy array
		
		:param func_name: какую функцию активации использовать, default to "sigmoid"
		:type func_name: str
		
		:rtype: numpy array
		:return: активированные значения нейронов слоя  
		"""

		output = [0]*len(array)

		if func_name in ("sigmoid", "tanh", "relu"):
			if func_name == "sigmoid":
				func = self.sigmoid
			elif func_name == "tanh":
				func = self.tanh
			elif func_name == "relu":
				func = self.relu
			
			#vfunc = np.vectorize(func)
			#return vfunc(array)
			return func(array)
		
		elif func_name in ("softmax",):
			func = self.softmax
			return func(array)

	
	def cross_entropy_loss(self, s, y):
		# https://mattpetersen.github.io/softmax-with-cross-entropy
		"""Return the cross-entropy of vectors y and s.

		:type y: ndarray
		:param y: one-hot vector encoding correct class

		:type s: ndarray
		:param s: softmax vector

		:returns: scalar cost
		"""
		# Naively computes log(s_i) even when y_i = 0
		# return -y.dot(np.log(s))
		
		# Efficient, but assumes y is one-hot
		return -np.log(s[np.where(y)])
		
		
	def calc_errors(self, output, func="MSE"):
		"""Подсчёт ошибок на последнем слое для отображения прогресса обучения
		
		:param output: значения последнего слоя
		:type output: numpy array
		
		:param func: какую функции ошибки использовать, default to "MSE"
		:type func: str
		
		:rtype: float
		:return: значени ошибки
		"""
		
		# MSE 
		if func == "MSE":
			errors = sum([0.5 *(self.out_true[idx] - o)**2 for idx, o in enumerate(output)])#/len(output)
		elif func == "CE":
			errors = self.cross_entropy_loss(output[None, ...], np.array(self.out_true)[None, ...])
		return errors


	def forward(self, mode="train"):
		"""Прямой проход по сети
		
		:param mode: режим работы сети - обучение или тест работы сети
		:type mode: str
		
		:rtype: float(mode="train") or tuple(mode="test")
		:return: значения ошибок на тренировке/обучении или последний слой
				 до и после активации 
		"""
		
		assert (len(self.layers[0]) == self.structure[0][0]), "количество входных параметров должно совпадать с количеством входных нейронов!"
		
		# скрытые слои
		for idx, layer in enumerate(self.layers[1:], start=1):
			
			#print("!"*20, self.layers[idx-1] , self.weights[idx-1], self.bias[idx-1])
			# print(idx, self.weights[idx-1].shape)
			cur_inp = (self.layers[idx-1] @ self.weights[idx-1].T) + self.bias[idx-1]
			cur_out = self.calc_output(cur_inp, self.structure[idx][2])	
			self.layers[idx] = cur_out # записывается активироанный слой
		
		if mode == "train":
			errors = self.calc_errors(self.layers[-1], error_func)
			return errors
		elif mode == "test":
			return cur_inp, cur_out
		
	
	#######################################
	#функции для обратного распространения#
	#######################################
	
	def derivative(self, x, func_name="sigmoid"):
		"""Подсчёт производных по слоям
		
		:param x: слой, для которого считается производная
		:type x: numpy array
		
		:param func_name: по какой функции берём производную
		:type func_name: str
		
		:rtupe: numpy array
		:return: значения прозводных по слою
		"""
		
		if func_name in ("sigmoid", "tanh", "relu"):
			if func_name == "sigmoid":
				return x*(1-x)
			elif func_name == "tanh":
				return 1 - x**2
			elif func_name == "relu":
				return (x > 0).astype(int)
							
		elif func_name in ("softmax",): 
			# https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
			# я даже понял
			# надо было сразу считать производные по весу от ОБЩЕЙ ошибки
			S_vector = self.layers[-1].reshape(self.layers[-1].shape[0], 1)
			#print(S_vector)
			S_matrix = np.tile(S_vector, self.layers[-1].shape[0])
			return np.diag(self.layers[-1]) - (S_matrix * np.transpose(S_matrix))
			
	
	# чтобы понять, откуда эти формулы, нужно на бумаге написать обратный проход для сети из входа, двух скрытых слоёв и выходного
	def calc_output_deltas(self, func="MSE"):
		"""Расчёт дельт для последнего слоя
		
		:param func: какая функция использовалась для активации слоя
		:type func: str
	
		:rtupe: numpy array
		:return: значения дельт последнего(выходного) слоя	
		"""
		
		if func == "MSE":
			first = (self.layers[-1] - self.out_true)
		elif func == "CE":
			first = - (np.array(self.out_true) / (self.layers[-1]))
		
		if self.structure[-1][2] in ("sigmoid", "tanh"):
					# для случая с MSE									# значения		 # функция, от которой берём производную
			return first * self.derivative(self.layers[-1], self.structure[-1][2])
		
		elif self.structure[-1][2] in ("softmax",):
			return first @ self.derivative(self.layers[-1], self.structure[-1][2])
		
		
	# если что, здесь скрытые слои идут в обратном порядке, при вызове функции стоит шаг -1
	def calc_hidden_deltas(self, layer_idx):
		"""Подсчёт дельт скрытых слоёв сети
		
		:param layer_idx: индекс скрытого слоя
		:type layer idx: int
		
		:rtype: numpy array
		:return: дельты скрытого слоя, нужные для вычисления модификаторов весов		
		"""
		#first = self.deltas[0] @ self.weights[layer_idx] # old
		first = self.deltas[0] @ (self.weights[layer_idx] - self.weights_store[layer_idx] * 0.9)  # пробуем сделать "заглядывание вперёд" по алгоритму Нестерова. надеюсь, я всё правильно понял.
		second = first * self.derivative(self.layers[layer_idx], self.structure[layer_idx][2])
		return second
		
		
	def calc_modificators(self, deltas, idx):
		"""Вычисление модификаторов для весов
		
		:param deltas: дельты слоя Б, который получился умножением определённых весов на слой А
		:type layer idx: numpy array
		
		:param idx: индекс слоя А
		:type idx: int
		
		:rtype: numpy array
		:return: модификаторы весов, на которые умножали слой А
		"""
		
		'''
		output = []
		for delta in deltas:
			# для вычисления модификатора дельта слоя должна быть умножена на значение нейрона предыдущего слоя
			# но раз у нас дельт на одну меньше(для входного не считаем) то индексы предыдущих слоёв можно брать и так
			output.append(delta * self.layers[idx])
		print(output) # это и следующие выражения идентичны
		print((deltas * self.layers[idx].reshape(-1, 1)).T)
		print(deltas[:, None] * self.layers[idx][None, :])
		return np.array(output)
		'''

		return (deltas * self.layers[idx].reshape(-1, 1)).T # здесь используются активированные значения слоя
		
		
	def update_weights(self):
		"""Обновление весов сети"""
		
		for idx, deltas in enumerate(self.deltas):
			mods = self.alpha * self.calc_modificators(deltas, idx)
			# self.weights[idx] = self.weights[idx] - mods# OLD
			# https://habr.com/ru/post/318970/
			v_next = self.weights_store[idx] * 0.9 + mods # начало объяснения алгоритма Нестерова
			self.weights_store[idx] = v_next			
			self.weights[idx] = self.weights[idx] - v_next # mods
			
	
	# обратное распространение ошибки
	# обычный градиентный спуск
	# считаем дельты
	def backward(self):
		"""Функция, вызов которой приведёт к подсчёту дельт всех слоёв"""
		# вызываем это из цикла обучения
		# сделал так, чтобы можно было использовать пакетный спуск
		# то есть сначала сохранить все дельты последнего слоя, потом найти среднее
		# и уже после искать дельты скрытых слоёв и модифицировать веса
		# upd: пока не использовать
		self.deltas.append(self.calc_output_deltas(error_func)) 
		
		for idx in range(self.num_hidden_layers, 0, -1):# исключая выходной слой
			self.deltas.insert(0, self.calc_hidden_deltas(idx))
	
## test
'''
# данные из статьи Matt Mazur про backpropagation
struct = ((2, 0.35, "sigmoid"), (2, 0.6, "sigmoid"), (2, 0, "sigmoid")) # кол-во нейронов, величина смещения
s = SimpleNet(struct)

# временно для тестов вставить эти веса
s.weights = [	
				   np.array([[0.15, 0.20], [0.25, 0.30]]), 
				   np.array([[0.40, 0.45], [0.50, 0.55]])]
				   
s.layers[0] = np.array([0.05, 0.1])
s.out_true = np.array([.01,  0.99])



for _ in range(1):
	print(s.forward()) # ошибка
	s.backward()
	s.update_weights()
	s.deltas.clear()

print(s.weights)

'''

# должны получится веса в тесте
#[[(0.1497807161327628, 0.19956143226552567), (0.24975114363236958, 0.29950228726473915)], [(0.35891647971788465, 0.4086661860762334), (0.5113012702387375, 0.5613701211079891)]]

if __name__ == "__main__":
	#'''
	# тренировка на 20к картинках MNIST
	data = open("mnist_train_small.csv", "r").readlines()
	len_data = len(data)
				#N  #bias #f.activ.
	struct = ((784, 0, "tanh"), (50, 0,"tanh"), (20, 0,"tanh"), (10, 0, "softmax"))	

	# загрузка весов
	weights_file = "weights_tanh_softmax.npz"

	net = SimpleNet(struct, alpha = 0.05, weights_file=None) # заменить None на weights_file для загрузки готовых весов

	mode = "train" # "test"

	n_iter = 2 # ставим 0, если сразу хотим тестить

	def create_true_vector_mnist(value):
		output = [0 for _ in range(10)]
		output[value] = 1
		return output


	for iter_idx in range(n_iter):
		shuffle(data)
		for idx, row in enumerate(data):
			out_true = create_true_vector_mnist(int(row[0]))
			net.out_true = out_true
			row = row.split(",")
			inp = np.array([int(s)/255 for s in row[1:]]) # нормализация
			net.layers[0] = inp
			
			errors = net.forward()
			if mode == "train":
				# делаю так, чтобы можно было использовать пакетный градиентный спуск
				net.deltas.insert(0, net.calc_output_deltas()) 
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
				print("epoch:", iter_idx, "current_index/data", f"{idx}/{len_data}", "true:", out_true.index(max(out_true)), "prediction:", net.layers[-1].tolist().index(max(net.layers[-1])) )
				print("error", errors, net.layers[-1])
	#'''
	if mode == "train":
		np.savez("weights_tanh_softmax", *net.weights)
