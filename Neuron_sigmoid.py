import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random

iris = datasets.load_iris()
iris_data = iris.data
sl_data = iris.data[:100,0]
sw_data = iris.data[:100,1]

sl_avg = np.average(sl_data)
sl_data -= sl_avg
sw_avg = np.average(sw_data)
sw_data -= sw_avg

train_data = []
for i in range(100):
    correct = iris_data.target[i]
    train_data.append([sl_data[i],sw_data[i],correct])

e = np.e
def sigmoid(x):
    return 1/(1/1 + e**-x)

class Neuron:
    def __init__(self):
        self.input_sum = 0.0
        self.output = 0.0

    def set_input(self,inp):
        self.input_sum += inp

    def get_output(self):
        self.output = sigmoid(self.input_sum)
        return self.output
    
    def reset(self):
        self.input_sum = 0
        self.output = 0
    
class NeuralNetwork:
    def __init__(self):
        self.w_im = [[4.0, 4.0],[4.0, 4.0],[4.0, 4.0]]
        self.w_o = [[1.0, -1.0, 1.0]]
        
        #バイアス
        self.b_m = [3.0, 0.0, -3.0]
        self.b_o = [-0.5]
        
        self.input_layer = [0.0, 0.0]
        self.middle_layer = [Neuron(),Neuron(),Neuron()]
        self.output_layer = [Neuron()]

    def commit(self,input_data):
        self.input_layer[0] = input_data[0]
        self.input_layer[1] = input_data[1]
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()
        self.middle_layer[2].reset()
        self.output_layer[0].reset()

        #出力層～中間層
        self.middle_layer[0].set_input(self.input_layer[0]*self.w_im[0][0])
        self.middle_layer[0].set_input(self.input_layer[1]*self.w_im[0][1])
        self.middle_layer[0].set_input(self.b_m[0])

        self.middle_layer[1].set_input(self.input_layer[0]*self.w_im[1][0])
        self.middle_layer[1].set_input(self.input_layer[1]*self.w_im[1][1])
        self.middle_layer[1].set_input(self.b_m[1])

        self.middle_layer[2].set_input(self.input_layer[0]*self.w_im[2][0])
        self.middle_layer[2].set_input(self.input_layer[1]*self.w_im[2][1])
        self.middle_layer[2].set_input(self.b_m[2])
        
        #中間層～出力層
        self.output_layer[0].set_input(self.middle_layer[0].get_output()*self.w_o[0][0])
        self.output_layer[0].set_input(self.middle_layer[1].get_output()*self.w_o[0][1])
        self.output_layer[0].set_input(self.middle_layer[2].get_output()*self.w_o[0][2])
        self.output_layer[0].set_input(self.b_o[0])
        return self.output_layer[0].get_output()
    
    def train(self,correct):
        k = 0.3
        output_m0 = self.middle_layer[0].output
        output_m1 = self.middle_layer[1].output
        output_m2 = self.middle_layer[2].output
        output_o = self.output_layer[0].output

        delta = (output_o - correct)* output_o* (1-output_o)
        delta_1 = delta* self.w_o[0][0]* output_o* (1-output_m0)
        delta_2 = delta* self.w_o[0][1]* output_m1* (1-output_m1)
        delta_3 = delta* self.w_o[0][2]* output_m2* (1-output_m2)

        self.w_o[0][0] -= k* delta* output_m0
        self.w_o[0][1] -= k* delta* output_m1
        self.w_o[0][2] -= k* delta* output_m2
        self.b_o[0] -= k* delta

        self.w_im[0][0] -= k* delta_1* self.input_layer[0]
        self.w_im[0][1] -= k* delta_1* self.input_layer[1]
        self.w_im[1][0] -= k* delta_2* self.input_layer[0]
        self.w_im[1][1] -= k* delta_2* self.input_layer[1]
        self.w_im[2][0] -= k* delta_3* self.input_layer[0]
        self.w_im[2][1] -= k* delta_3* self.input_layer[1]
        self.b_m[0] -= k* delta_1
        self.b_m[1] -= k* delta_2
        self.b_m[2] -= k* delta_3

neural_network = NeuralNetwork()

def show_graph(epoch):
    print("Epoch:",epoch)

    st_predicted = [[],[]]
    vc_predicted = [[],[]]
    for data in train_data:
        if neural_network.commit(data) < 0.5:
           st_predicted[0].append(data[0]+sl_avg)
           st_predicted[1].append(data[1]+sw_avg)
        else:
           vc_predicted[0].append(data[0]+sl_avg)
           vc_predicted[1].append(data[1]+sw_avg)

           plt.scatter(st_predicted[0],st_predicted[1],label="Setosa")
           plt.scatter(vc_predicted[0],vc_predicted[1],label="Versicolor")
           plt.legend()

           plt.xlabel("Sepal length (cm)")
           plt.ylabel("Sepal width (cm)")
           plt.title("Predicted")
           plt.show()

show_graph(0)

for t in range(0,64):
    random.shuffle(train_data)
    for data in train_data:
        neural_network.commit(data[:2])
        neural_network.train(data[2])
    if t+1 in [1, 2, 4, 8, 16, 32, 64]:
        show_graph(t+1)