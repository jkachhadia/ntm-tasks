import theano
import numpy as np
import random as rn

dc={
    '0':[0,0,0,1],
    '1':[1,0,0,0],
    '2':[0,1,0,0],
    '3':[0,0,1,0],
    '-':[0,0,0,0]
}

dcs={
    '0':[1,0],
    '1':[0,1],
    '-':[0,0]
}

def get_train(n):
    xtrain=[]
    ytrain=[]
    for _ in range(n):
        xtrain.append(rn.randint(1,3))
    xtrain.append(0)
    for _ in range(n):
        xtrain.append('-')
    xtrain.append(0)
    for _ in range(n):
        ytrain.append('-')
    ytrain.append(0)
    ytrain.append(xtrain[0])
    for x in range(1,n):
        if xtrain[x] in xtrain[:x]:
            ytrain.append('-')
        else:
            ytrain.append(xtrain[x])
    ytrain.append(0)
    ytrain[0]=1
    x=[]
    y=[]
    for l in range(len(xtrain)):
        x.append(dc[str(xtrain[l])])
        y.append(dc[str(ytrain[l])])
    x=np.array([x])
    y=np.array([y])

    # xt.append([x])
    # yt.append([y])
    return x,y
def add(a,b):
    if a==1 and b==1:
        return 0
    elif a==1 and b==0:
        return 1
    elif a==0 and b==1:
        return 1
    else:
        return 0

def get_strain(n):
    x=[]
    y=[]
    for _ in range(n):
        x.append(rn.randint(0,1))
    x.append('-')
    for _ in range(n):
        x.append(rn.randint(0,1))
    x.append('-')
    for _ in range(n):
        y.append('-')
    y.append('-')
    for a,b in zip(x[:n],x[n+1:-1]):
        y.append(add(a,b))
    y.append('-')
    xt=[]
    yt=[]
    for l in range(len(x)):
        xt.append(dcs[str(x[l])])
        yt.append(dcs[str(y[l])])
    xt=np.array([xt])
    yt=np.array([yt])
    return xt,yt




class Task(object):

    def __init__(self, max_iter=None, batch_size=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            params = self.sample_params()
            return (self.num_iter - 1), self.sample(**params)
        else:
            raise StopIteration()

    def sample_params(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()


class UnTask(Task):

    def __init__(self, size, max_length, min_length=1, max_iter=None, \
        batch_size=1, end_marker=False):
        super(UnTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
        self.size = size
        self.min_length = min_length
        self.max_length = max_length
        self.end_marker = end_marker

    def sample_params(self, length=None):
        length=4
        if length is None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        return {'length': length}

    def sample(self, length):
        example_input,example_output=get_train(4,)

        return example_input, example_output

class SumTask(Task):
	
	    def __init__(self, max_length, min_length=1, max_iter=None, \
	                 batch_size=1, end_marker=False):
	        super(SumTask, self).__init__(max_iter=max_iter, batch_size=batch_size)
	        self.min_length = min_length
	        self.max_length = max_length
	        self.end_marker = end_marker
	
	    def sample_params(self, length=4):
	        if length is None:
	            length = np.random.randint(self.min_length, self.max_length + 1)
	        return {'length': length}
	
	    def sample(self, length):
	        num1 = np.random.binomial(1, 0.5, (self.batch_size, 1, length))
	        num2 = np.random.binomial(1, 0.5, (self.batch_size, 1, length))
	        example_input = np.zeros((self.batch_size, 3 * length + 3 + self.end_marker, \
	            3), dtype=theano.config.floatX)
	        example_output = np.zeros((self.batch_size, 3 * length + 3 + self.end_marker, \
	            3), dtype=theano.config.floatX)
	        sum_array = np.zeros((self.batch_size, 1, length + 1), \
	                             dtype=theano.config.floatX)
	        for i in range(self.batch_size):
	            rem = 0
	            for j in range(length):
	                s = num1[i][0][j] + num2[i][0][j] + rem
	                sum_array[i][0][j] = s % 2
	                rem = int(s) / 2
	            sum_array[i][0][length] = rem
	        example_input[:, :length, 0] = num1
	        example_input[:, length, 1] = 1
	        example_input[:, (length + 1):(2 * length + 1), 0] = num2
	        example_input[:, (2 * length + 1), 2] = 1
	        example_output[:, (2 * length + 2):(3 * length + 3), 0] = sum_array
	        if self.end_marker:
	            example_output[:, (3 * length + 3), 2] = 2
	
	        return example_input, example_output