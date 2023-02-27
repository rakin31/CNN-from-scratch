import csv
import os
import math
from math import floor
import sklearn
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

np.random.seed(42)
import pickle as pkl
import sys
import seaborn as sns
import matplotlib.pyplot as plt





# Classes

# Convolution
class Convolution:
    def __init__(self, num_of_out_channels, filter_dim, stride, padding, input_channel):
        # self.in_channels = in_channels
        self.num_of_out_channels = num_of_out_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel

        # initializing weights and bias
        self.weight = np.random.randn(self.num_of_out_channels, input_channel, self.filter_dim, self.filter_dim) * math.sqrt(2 / (self.num_of_out_channels * self.filter_dim * self.filter_dim))
        self.bias = np.zeros(self.num_of_out_channels)

        # initializing cache
        self.cache = None


    def createWindowsforconv(self, input, out_size, k_size, padding, stride, dilate):
        inp = input
        pad = padding

        if dilate == 0:
            inp = input
        else:
            inp = np.insert(inp, range(1, input.shape[2]), 0, axis=2)
            inp = np.insert(inp, range(1, input.shape[3]), 0, axis=3)


        if pad != 0:
            inp = np.pad(inp, pad_width=((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=(0.,))

        input_batch, input_channel, outHeight, outWeight = out_size
        output_batch, output_channel, _n, _n = input.shape
        b_strides, c_strides, kern_height_strides, kern_width_strides = inp.strides

        # print("inp.shape", inp.shape)
        # print("out_size", out_size)
        # print("k_size", k_size)
        wdw = np.lib.stride_tricks.as_strided(inp, (output_batch, output_channel, outHeight, outWeight, k_size, k_size), (b_strides, c_strides, stride * kern_height_strides, stride * kern_width_strides, kern_height_strides, kern_width_strides))
        return wdw

    def forward(self, input_tensor):
        # print("in convolution_layer")
        n, channel, height, weight = input_tensor.shape
        outputHeight = int((height - self.filter_dim + 2 * self.padding) / self.stride) + 1
        outputWidth = int((weight - self.filter_dim + 2 * self.padding) / self.stride) + 1

        windowsforConv = self.createWindowsforconv(input_tensor, (n, channel, outputHeight, outputWidth), self.filter_dim, self.padding, self.stride, 0)

        out = np.einsum('bihwkl,oikl->bohw', windowsforConv, self.weight)
        out = out + self.bias[None, :, None, None]

        # print("out.shape", out.shape)
        # print(out)
        self.cache = input_tensor, windowsforConv

        return out

    def backward(self, doutput, learning_rate):
        # print("in convolution_layer")
        # print("d_output.shape", doutput.shape)
        # print(doutput)
        x, windows = self.cache
        # print("x_shape", x.shape)

        if self.padding == 0:
            padding = self.filter_dim - 1
        else:
            padding = self.padding

        doutput_windows = self.createWindowsforconv(doutput, doutput.shape, self.filter_dim, padding=padding, stride=1, dilate=self.stride - 1)
        rottated_kernel = np.rot90(self.weight, 2, axes=(2, 3))

        db = np.sum(doutput, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', windows, doutput)
        dx = np.einsum('bohwkl,oikl->bihw', doutput_windows, rottated_kernel)

        # print("dx.shape", dx.shape)
        # print(dx)
        # print("dw.shape", dw.shape)
        # print(dw)
        # print("db.shape", db.shape)
        # print(db)

        self.weight = self.weight - (dw * learning_rate)
        self.bias = self.bias - (db * learning_rate)

        return dx


# ReLU
class ReLU:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, doutput, learning_rate):
        if self.input_tensor is None:
            return None
        else:
            dZ = np.array(doutput, copy=True)
            # print("dZ.shape", dZ.shape)
            dZ[self.input_tensor <= 0] = 0
            # print(dZ)
        return dZ


# Max Pooling
class maxPool:
    def __init__(self, filter_dim, stride, padding):
        self.pool_size = (filter_dim, filter_dim)
        self.filter_dim = filter_dim
        self.stride = stride
        self.input_tensor = None
        self.max_indices = None
        self.padding = padding
        self.cache = {}


    def forward(self, input_tensor, training=True):
        n, channel, height, width = input_tensor.shape
        # print("in maxpool")
        # print("input_tensor.shape", input_tensor.shape)
        height_poolwindow, width_poolwindow = self.pool_size
        out_height = int((height - height_poolwindow) / self.stride) + 1
        out_width = int((width - width_poolwindow) / self.stride) + 1

        windows = np.lib.stride_tricks.as_strided(input_tensor,shape=(n, channel, out_height, out_width, *self.pool_size), strides=(input_tensor.strides[0], input_tensor.strides[1], self.stride * input_tensor.strides[2], self.stride * input_tensor.strides[3], input_tensor.strides[2], input_tensor.strides[3]))
        # print("windows.shape", windows.shape)

        out = np.max(windows, axis=(4, 5))
        maxs = out.repeat(2, axis=2).repeat(2, axis=3)
        input_window = input_tensor[:, :, :out_height * self.stride, :out_width * self.stride]
        mask = np.equal(input_window, maxs).astype(int)

        if training == True:
            self.cache['X'] = input_tensor
            self.cache['mask'] = mask

        return out


    def backward(self, doutput, learning_rate):
        input_tensor = self.cache['X']
        # print("in_back")
        # print(input_tensor.shape)
        mask = self.cache['mask']
        height_poolwindow, width_poolwindow = self.pool_size
        # print("mask.shape", mask.shape)

        dA = doutput.repeat(height_poolwindow, axis=2).repeat(width_poolwindow, axis=3)
        # print("dA.shape", dA.shape)
        dA = np.multiply(dA, mask)
        # print(dA)
        pad = np.zeros(input_tensor.shape)
        # print("pad", pad)
        pad[:, :, :dA.shape[2], :dA.shape[3]] = dA

        return pad

# Flattening Layer
class Flattening_Layer:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        flattened = input_tensor.reshape(input_tensor.shape[0], -1)
        return flattened

    def backward(self, doutput, learning_rate):
        rtn_val = doutput.reshape(self.input_tensor.shape)
        return rtn_val

# Fully Connected Layer
class FullyConnected_Layer:
    def __init__(self, input_dimension, output_dimension):
        self.input_tensor = None
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = np.random.randn(self.output_dimension, self.input_dimension)*np.sqrt(2/self.input_dimension)
        self.bias = np.zeros((self.output_dimension, 1))

    def forward(self, input_tensor):
        # print("in fully_connected_layer")
        # print("input_tensor.shape", input_tensor.shape)
        # print("self.weights.shape", self.weights.shape)
        # print("self.bias.shape", self.bias.shape)
        self.input_tensor = input_tensor
        x = np.dot(self.weights, self.input_tensor.T)
        x = x + self.bias
        rtn_val = x.T
        return rtn_val


    def backward(self, doutput, learning_rate):
        # print("in fully_connected_layer")
        # print("d_output.shape", gradient.shape)
        # print("self.input_tensor.shape", self.input_tensor.shape)

        # intializing the gradients
        dX=np.zeros(self.input_tensor.shape)
        dW=np.zeros(self.weights.shape)
        dB=np.zeros(self.bias.shape)

        dX=np.dot(doutput, self.weights)
        dW=np.dot(doutput.T, self.input_tensor)
        dB=np.sum(doutput.T, axis=1, keepdims=True)

        self.weights = self.weights - (dW*learning_rate)
        self.bias = self.bias - (dB*learning_rate)

        rtn_val = dX.reshape(self.input_tensor.shape)

        return rtn_val

# Softmax Layer
class Softmax:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        exp_input = np.exp(input_tensor)
        exp_sum = np.sum(exp_input, axis=1, keepdims=True)
        self.softmax = exp_input / exp_sum

        return self.softmax

    def backward(self, doutput, learning_rate):
        return doutput


# Lenet5

class LeNetModel:
    def __init__(self, dimensions):
        self.conv1 = Convolution(6, 5, 1, 2, 3)
        stride = 1
        pad = 2
        ker = 5
        # dimensions = floor((dimensions - 5 + (2 * 2)) / 1) + 1
        dimensions = floor((dimensions - ker + (2 * pad)) / stride)
        dimensions = dimensions + 1
        self.relu1 = ReLU()
        self.maxpool1 = maxPool(2, 2, 0)
        ker_m = 2
        stride_m = 2
        # dimensions = floor((dimensions - 2) / 2) + 1
        dimensions = floor((dimensions - ker_m) / stride_m)
        dimensions = dimensions + 1
        self.conv2 = Convolution(16, 5, 1, 2, 6)
        ker2 = 5
        pad2 = 2
        stride2 = 1
        # dimensions = floor((dimensions - 5 + (2 * 2)) / 1) + 1
        dimensions = floor((dimensions - ker2 + (2 * pad2)) / stride2)
        dimensions = dimensions + 1
        self.relu2 = ReLU()
        self.maxpool2 = maxPool(2, 2, 0)
        ker_m2 = 2
        stride_m2 = 2
        dimensions = floor((dimensions - ker_m2) / stride_m2)
        dimensions = dimensions + 1
        channels = 16
        self.flatten = Flattening_Layer()
        in_channel = dimensions * dimensions * channels
        # print(in_channel)
        self.fc1 = FullyConnected_Layer(in_channel, 120)
        self.relu3 = ReLU()
        self.fc2 = FullyConnected_Layer(120, 84)
        self.relu4 = ReLU()
        self.fc3 = FullyConnected_Layer(84, 10)
        self.softmax = Softmax()

    def forward(self, input_tensor):
        x = self.conv1.forward(input_tensor)
        # print("conv1 output shape: ", x.shape)
        x = self.relu1.forward(x)
        # print("relu1 output shape: ", x.shape)
        x = self.maxpool1.forward(x)
        # print("maxpool1 output shape: ", x.shape)
        x = self.conv2.forward(x)
        # print("conv2 output shape: ", x.shape)
        x = self.relu2.forward(x)
        # print("relu2 output shape: ", x.shape)
        x = self.maxpool2.forward(x)
        # print("maxpool2 output shape: ", x.shape)
        x = self.flatten.forward(x)
        # print("flatten output shape: ", x.shape)
        x = self.fc1.forward(x)
        # print("fc1 output shape: ", x.shape)
        x = self.relu3.forward(x)
        # print("relu3 output shape: ", x.shape)
        x = self.fc2.forward(x)
        # print("fc2 output shape: ", x.shape)
        x = self.relu4.forward(x)
        # print("relu4 output shape: ", x.shape)
        x = self.fc3.forward(x)
        # print("fc3 output shape: ", x.shape)
        x = self.softmax.forward(x)
        # print("softmax output shape: ", x.shape)
        # row_sum = x.sum(axis=1)
        # print(row_sum)
        return x

    def backward(self, doutput, learning_rate):
        d_x = self.softmax.backward(doutput, learning_rate)
        d_x = self.fc3.backward(d_x, learning_rate)
        d_x = self.relu4.backward(d_x, learning_rate)
        d_x = self.fc2.backward(d_x, learning_rate)
        d_x = self.relu3.backward(d_x, learning_rate)
        d_x = self.fc1.backward(d_x, learning_rate)
        d_x = self.flatten.backward(d_x, learning_rate)
        d_x = self.maxpool2.backward(d_x, learning_rate)
        d_x = self.relu2.backward(d_x, learning_rate)
        d_x = self.conv2.backward(d_x, learning_rate)
        d_x = self.maxpool1.backward(d_x, learning_rate)
        d_x = self.relu1.backward(d_x, learning_rate)
        d_x = self.conv1.backward(d_x, learning_rate)
        return d_x

# cross_entropy_loss
def find_cross_entropy_loss(y_predicted, y_actual):
    loss = np.sum(-np.log(y_predicted) * y_actual)
    return loss


# file_reading
def read_images_create_matrix(y_info, start, end):
    # images = Path(folder_path).glob('*.png')
    # print(images)
    matrices = []
    all_labels = []
    for i in range(start, end):
        filename, label = y_info[i]
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            img = cv2.imread(filename)
            # print(img.shape)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            # print(img.shape)
            matrix = np.array(img)
            # matrix = matrix/255
            matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB)
            matrix = matrix / 255
            # print(matrix.shape)
            matrices.append(matrix)
            all_labels.append(label)
            # print(matrix)
            # img.close()
    return matrices, all_labels





# test
def test(y_info, pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        my_model = pkl.load(f)
    x_train, y_train = read_images_create_matrix(y_info, 0, len(y_info))
    y_real = y_train
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_train_reshaped = np.transpose(x_train, (0, 3, 1, 2))
    y_prediction = my_model.forward(x_train_reshaped)
    # print("y_prediction : ", y_prediction)
    predicted_label = np.argmax(y_prediction, axis=1)
    ohl = np.zeros((len(y_real), 10))
    ohl[np.arange(len(y_real)), y_real] = 1
    loss = find_cross_entropy_loss(y_prediction, ohl)/len(y_train)
    acc = sklearn.metrics.accuracy_score(y_train, predicted_label)
    F1_score = f1_score(y_train, predicted_label, average='macro')
    print("accuracy : ", acc)
    print("loss : ", loss)
    print("F1_score : ", F1_score)
    return predicted_label
    # with open()


if __name__ == '__main__':
    if __name__ == '__main__':
        # matrices = []
        # x_train = read_images_create_matrix_2('./training-a/')
        # print(len(x_train))
        # x_train = read_images_create_matrix_2('./training-b/', x_train)
        # x_train = read_images_create_matrix_2('./training-c/', x_train)
        # x_train = np.array(x_train)
        # x_train_reshaped = np.transpose(x_train, (0, 3, 1, 2))
        # print(x_train_reshaped.shape)
        pkl_file_path = sys.argv[1]
        folder_name = './training-d/'
        df = pd.read_csv('training-d.csv')
        y_test_digit = df['digit'].values
        y_test_filename = df['filename'].values
        y_info_test = []
        for i in range(len(y_test_filename)):
            img_path = os.path.join(folder_name, y_test_filename[i])
            y_info_test.append((img_path, y_test_digit[i]))
        # print(y_info)
        pr_label = test(y_info_test, pkl_file_path)
        print()
        print()
        print("============================================")
        print("Test done\n\n")

        filename_out_csv = "1705026_prediction.csv"
        fields = ['Filename', 'Digit']
        csvfile = open(filename_out_csv, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for i in range(len(y_info_test)):
            csvwriter.writerow([y_test_filename[i], pr_label[i]])
        csvfile.close()
        confusion = confusion_matrix(y_test_digit,pr_label)
        sns.heatmap(confusion, annot=True, fmt='d',  cbar=False, xticklabels=[0,1,2,3,4,5,6,7,8,9], yticklabels=[0,1,2,3,4,5,6,7,8,9])

        # Add labels and title
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        # Show the plot
        plt.show()

