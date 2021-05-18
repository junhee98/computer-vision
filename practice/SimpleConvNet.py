# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:18:06 2021

@author: taeso
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from dataset.mnist import load_mnist
from common.layers import *
from common.gradient import numerical_gradient #수치미분함수
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split
import imageload as il

class SimpleConvNet:
    """
    Parameters
    -----------
    input_size : 입력 크기(MNIST의 경우엔 784)
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트(e.g. [100, 100, 100])
    output_size : 출력 크기 (MNIST의 경우엔 10)
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정 (e.g. 0.01)
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        
        #가중치 초기화
        self.params = {}
        self.params['W1'] = np.sqrt(1/input_dim[1]) * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = np.sqrt(1/pool_output_size) * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.sqrt(1/hidden_size) * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        #계층생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
        self.last_layer = SoftmaxWithLoss()
        
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
            
        return x
    
    
    def loss(self, x, t):
        """손실함수를 구한다.
        
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i+1) * batch_size]
            tt = t[i * batch_size:(i+1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
            
        return acc / x.shape[0]
    
    
    def numerical_gradient(self, x, t):
        """기울기를 구한다 (수치미분)
        
        Parameter
        ----------
        x : 입력 데이터
        t : 정답 레이블
        
        
        Return
        --------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1'], grads['W2'], ... 각 층의 가중치
            grads['b1'], grads['b2'], ... 각 층의 편향
        """
        
        loss_w = lambda w: self.loss(x, t)
        
        grads = {}
        for idx in (1, 2, 3):
            grads['W'+ str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b'+ str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
            
        return grads
    
    
    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법)
        
        Parameter
        ---------
        x : 입력 데이터
        t : 정답 레이블
        
        Return
        --------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1'], grads['W2'], ... 각 층의 가중치
            grads['b1'], grads['b2'], ... 각 층의 편향
        """
        #foward
        self.loss(x, t)
        
        #backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        #결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads
        
    
    #사용한 파라미터를 저장한다
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
                
                
                
                
    def load_params(self, file_name="params,pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
            
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
            
            
            
            
            
#데이터 읽기
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
x_train, x_test, t_train, t_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=1000)

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('int32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('int32')
t_train = t_train.astype('int32')
t_test = t_test.astype('int32')

#28x28 image 불러오기
mypath='./raw-img/cane/'

loadedImage = il.imageLoad()
images = np.empty(len(loadedImage), dtype=object) #file pixel info
    
#image 정보 출력
for n in range(0, len(loadedImage)):
  images[n] = cv2.imread( join(mypath,loadedImage[n]) )
  print(images[n])


max_eporchs = 20

network = SimpleConvNet()


#매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")


#하이퍼파라미터
iters_num = 10000 # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0] # 60000개
batch_size = 100 #미니배치 크기
learning_rate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []



#1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch) #600

for i in range(iters_num): #10000
    # 미니배치 획득  # 랜덤으로 100개씩 뽑아서 10000번을 수행하니까 백만번
    batch_mask = np.random.choice(train_size, batch_size) #100개 씩 뽑아서 10000번 백만번
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    
    
    #기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
        
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss) #cost가 점점 줄어드는것을 보려고
    #1에폭당 정확도 계산 # 여기는 훈련이 아니라 1에폭 되었을 때 정확도만 체크
    
    if i % iter_per_epoch == 0: #600번마다 정확도 쌓는다
        print(x_train.shape) #60000,784
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc) # 10000/600 개 16개  #정확도가 점점 올라감
        test_acc_list.append(test_acc) # 10000/600 개 16개  #정확도가 점점 올라감
        print("train acc, test acc | " +str(train_acc) + ", " + str(test_acc))
    
    
    
    
#그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.aranger(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.xlabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
