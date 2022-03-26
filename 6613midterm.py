import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) =fashion_mnist.load_data()
print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
np.set_printoptions(threshold=np.inf)  
def set_label(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result
def img2col(x, ksize, stride):
    wx, hx, cx = x.shape              # [width,height,channel]
    feature_w = (wx - ksize) // stride + 1   # feature map size       
    image_col = np.zeros((feature_w*feature_w, ksize*ksize*cx))
    num = 0
    for i in range(feature_w):
        for j in range(feature_w): 
            image_col[num] =  x[i*stride:i*stride+ksize, j*stride:j*stride+ksize, :].reshape(-1)
            num += 1
    return image_col
    

## nn
class Linear(object):
    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel/2)
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale
        self.b = np.random.standard_normal(outChannel) / scale
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        self.x = x
        x_forward = np.dot(self.x, self.W) + self.b
        return x_forward

    def backward(self, delta, learning_rate):
        ## gradient calc
        batch_size = self.x.shape[0]
        self.W_gradient = np.dot(self.x.T, delta) / batch_size  # bxin bxout
        self.b_gradient = np.sum(delta, axis=0) / batch_size 
        delta_backward = np.dot(delta, self.W.T)                # bxout inxout
        ## backprop
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate 

        return delta_backward

## conv
class Conv(object):
    def __init__(self, kernel_shape, stride=1, pad=0):
        width, height, in_channel, out_channel = kernel_shape
        self.stride = stride
        self.pad = pad
        scale = np.sqrt(3*in_channel*width*height/out_channel)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(out_channel) / scale
        self.k_gradient = np.zeros(kernel_shape)
        self.b_gradient = np.zeros(out_channel)

    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), 'constant')
        bx, wx, hx, cx = self.x.shape
        wk, hk, ck, nk = self.k.shape             # kernel size
        feature_w = (wx - wk) // self.stride + 1  
        feature = np.zeros((bx, feature_w, feature_w, nk))
                       
        self.image_col = []
        kernel = self.k.reshape(-1, nk)
        for i in range(bx):
            image_col = img2col(self.x[i], wk, self.stride)                       
            feature[i] = (np.dot(image_col, kernel)+self.b).reshape(feature_w,feature_w,nk)
            self.image_col.append(image_col)
        return feature

    def backward(self, delta, learning_rate):
        bx, wx, hx, cx = self.x.shape # batch,14,14,inchannel
        wk, hk, ck, nk = self.k.shape # 5,5,inChannel,outChannel
        bd, wd, hd, cd = delta.shape  # batch,10,10,outChannel

        # self.k_gradient,self.b_gradient
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
        self.k_gradient /= bx
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= bx    

        # delta_backward
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0,1))      
        k_180 = k_180.swapaxes(2, 3)
        k_180_col = k_180.reshape(-1,ck)

        if hd-hk+1 != hx:
            pad = (hx-hd+hk-1) // 2
            pad_delta = np.pad(delta, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = img2col(pad_delta[i], wk, self.stride)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx,hx,ck)

        # backprop
        self.k -=  self.k_gradient * learning_rate
        self.b -=  self.b_gradient * learning_rate

        return delta_backward

## pool
class Pool(object):
    def forward(self, x):
        b, w, h, c = x.shape
        feature_w = w // 2
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))   # Record the location information of the maximum value during max pooling for backpropagation
        for bi in range(b):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        feature[bi, i, j, ci] = np.max(x[bi,i*2:i*2+2,j*2:j*2+2,ci])
                        index = np.argmax(x[bi,i*2:i*2+2,j*2:j*2+2,ci])
                        self.feature_mask[bi, i*2+index//2, j*2+index%2, ci] = 1                    
        return feature

    def backward(self, delta):
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask

## Relu
class Relu(object):        
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, delta):
        delta[self.x<0] = 0
        return delta

## Softmax
class Softmax(object):
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)
        for i in range(batchsize):
            predict_tmp = predict[i] - np.max(predict[i])
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax

def train(X_train, y_train):
    '''
    the training process was happen in my Ubuntu VM as its faster
    therefore the loading path is not in google colab format
    '''
   
    X_train = X_train.reshape(60000, 28, 28, 1) / 255.   # Input vector processing
    y_train = set_label(y_train, 60000) # create label (60000, 10) 
    
    conv1 = Conv(kernel_shape=(5,5,1,6))# 24x24x6
    relu1 = Relu()
    pool1 = Pool()# 12x12x6
    conv2 = Conv(kernel_shape=(5,5,6,16))# 8x8x16
    relu2 = Relu()
    pool2 = Pool()# 4x4x16
    nn = Linear(256, 10)
    softmax = Softmax()
   
    lr = 0.01
    batch = 100
    for epoch in range(10):        
        for i in range(0, 60000, batch):
            X = X_train[i:i+batch]
            Y = y_train[i:i+batch]

            predict = conv1.forward(X)
            predict = relu1.forward(predict)
            predict = pool1.forward(predict)
            predict = conv2.forward(predict)
            predict = relu2.forward(predict)
            predict = pool2.forward(predict)
            predict = predict.reshape(batch, -1) #flat
            predict = nn.forward(predict)

            loss, delta = softmax.cal_loss(predict, Y)

            delta = nn.backward(delta, lr)
            delta = delta.reshape(batch,4,4,16)
            delta = pool2.backward(delta)
            delta = relu2.backward(delta)
            delta = conv2.backward(delta, lr)
            delta = pool1.backward(delta)
            delta = relu1.backward(delta)
            conv1.backward(delta, lr)

            if i%90 == 0: #progress checking
              print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        lr *= 0.95**(epoch+1)
        k1=conv1.k, b1=conv1.b, k2=conv2.k, b2=conv2.b, w3=nn.W, b3=nn.b
        
def test():

    
    #X_test [10000,28,28]
    #y_test [10000]

    X_test = X_test.reshape(10000, 28, 28, 1) / 255.

    conv1 = Conv(kernel_shape=(5, 5, 1, 6))  # 24x24x6
    relu1 = Relu()
    pool1 = Pool()  # 12x12x6
    conv2 = Conv(kernel_shape=(5, 5, 6, 16))  # 8x8x16
    relu2 = Relu()
    pool2 = Pool()  # 4x4x16
    nn = Linear(256, 10)
    softmax = Softmax()

    conv1.k = k1
    conv1.b = b1
    conv2.k = k2
    conv2.b = b2
    nn.W = w3
    nn.b = b3

    num = 0
    for i in range(10000):
        X = X_test[i]
        X = X[np.newaxis, :] # increase dimension 
        Y = y_test[i]

        predict = conv1.forward(X)
        predict = relu1.forward(predict)
        predict = pool1.forward(predict)
        predict = conv2.forward(predict)
        predict = relu2.forward(predict)
        predict = pool2.forward(predict)
        predict = predict.reshape(1, -1) #flat
        predict = nn.forward(predict)

        predict = softmax.predict(predict)
        if i%500 == 0: #progress checking
          print('Testing ',i,'th in progress')
        if np.argmax(predict) == Y:
            num += 1
    accuracy = num/10000*100
    print("TEST-ACC: ", str(accuracy), "%")
    return accuracy
if __name__ == '__main__':
    train(X_train, y_train)
    acc = test(X_test, y_test)
    print("TEST-ACC: ", str(acc), "%")
