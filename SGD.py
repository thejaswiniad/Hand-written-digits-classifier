#Handwritten classifier using the MNSIT dataset without using machine learning libraries 
import random
import numpy as np
import sys
def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
if (len(sys.argv)>1):
    print("taking from commandline--")
    filename1 = sys.argv[1]
    train_image = np.loadtxt(filename1, delimiter=",")
    filename2 = sys.argv[2]
    train_label = np.loadtxt(filename2, delimiter=",")
    filename3 = sys.argv[3]
    test_image = np.loadtxt(filename3, delimiter=",")
else:
    print("taking from work directory--")
    train_image = np.loadtxt("train_image_new.csv", delimiter=",")
    train_label = np.loadtxt("train_label_new.csv", delimiter=",")
    test_image = np.loadtxt("test_image.csv", delimiter=",")

train_image = train_image / 255.0
test_image = test_image / 255.0

train_label=train_label.astype(int)
train_label = [vectorized_result(y) for y in train_label]

train_image = [np.reshape(x, (784, 1)) for x in train_image]
test_image = [np.reshape(x, (784, 1)) for x in test_image]


training_data = zip(train_image, train_label)



class ANN(object):

    def __init__(self, sizes):

        self.no_of_layers = len(sizes)
        self.sizes = sizes
        self.b1 = np.random.randn(100, 1)
        self.w1 = np.random.randn(100, 784) / np.sqrt(784)
        self.b2 = np.random.randn(10,1)
        self.w2 = np.random.randn(10, 100) / np.sqrt(100)

    def predict_test_values(self, a):

        a1 = sigmoid_fn(np.dot(self.w1, a)+self.b1)
        a2=softmax_fn(np.dot(self.w2,a1)+self.b2)
        return a2

    def Stochastic_grad_descent(self, training_data, epochs, mini_batch_size, eta,lmbda,test_data):
        training_data=list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.reform_bias_weight(mini_batch, eta,lmbda,n)
        pr=self.evaluate(test_data)
        np.savetxt("test_predictions.csv",pr,delimiter=",",fmt="%d")

    def reform_bias_weight(self, mini_batch, eta, lmbda, n):

        nabla_b1 = np.zeros(self.b1.shape)
        nabla_w1= np.zeros(self.w1.shape)
        nabla_b2 = np.zeros(self.b2.shape)
        nabla_w2= np.zeros(self.w2.shape)
        for x, y in mini_batch:
            delta_nabla_b1, delta_nabla_w1,delta_nabla_b2, delta_nabla_w2 = self.back_propogation(x, y)
            nabla_b1 = nabla_b1 + delta_nabla_b1
            nabla_w1 = nabla_w1 + delta_nabla_w1
            nabla_b2 = nabla_b2 + delta_nabla_b2
            nabla_w2 = nabla_w2 + delta_nabla_w2
        self.w1 = (1 - eta * (lmbda / n)) * self.w1 - (eta / len(mini_batch)) * nabla_w1
        self.b1 = self.b1 - (eta / len(mini_batch)) * nabla_b1
        self.w2 = (1 - eta * (lmbda / n)) * self.w2 - (eta / len(mini_batch)) * nabla_w2
        self.b2 = self.b2 - (eta / len(mini_batch)) * nabla_b2

    def back_propogation(self, x, y):

        update_b1 = np.zeros(self.b1.shape)
        update_w1= np.zeros(self.w1.shape)
        update_b2 = np.zeros(self.b2.shape)
        update_w2 = np.zeros(self.w2.shape)
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        z1 = np.dot(self.w1, activation) + self.b1
        zs.append(z1)
        activation1 = sigmoid_fn(z1)
        activations.append(activation)

        z2=np.dot(self.w2,activation1) + self.b2
        zs.append(z2)
        activation2 = softmax_fn(z2)
        activations.append(activation2)
        # backward pass
        cross_entropy_loss = activation2- y
        update_b2 = cross_entropy_loss
        update_w2 = np.dot(cross_entropy_loss, activation1.transpose())

        for l in range(2, self.no_of_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            cross_entropy_loss = np.dot(self.w2.transpose(), cross_entropy_loss) * sp
            update_b1 = cross_entropy_loss
            update_w1 = np.dot(cross_entropy_loss, activations[-l - 1].transpose())
        return update_b1, update_w1,update_b2,update_w2

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict_test_values(x)))for x in test_data]
        return test_results

def sigmoid_fn(z):
    return 1.0/(1.0+np.exp(-z))

def softmax_fn(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid_derivative(z):
    return sigmoid_fn(z)*(1-sigmoid_fn(z))

ann=ANN([784,100,10])

ann.Stochastic_grad_descent(training_data, 30, 10, 0.5, lmbda=5.0,test_data=test_image)


