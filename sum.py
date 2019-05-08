import theano
import theano.tensor as T
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init

from ntm.layers import NTMLayer
from ntm.memory import Memory
from ntm.controllers import DenseController
from ntm.heads import WriteHead, ReadHead
from ntm.updates import graves_rmsprop

from utils.generators import SumTask
from utils.visualization import Dashboard,learning_curve
from sklearn.metrics import log_loss
import json

def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

# from generate import get_train


def model(input_var, batch_size=1, size=1, num_units=100, memory_shape=(128, 20)):

    # Input Layer
    l_input = InputLayer((batch_size, None, size + 1), input_var=input_var)
    _, seqlen, _ = l_input.input_var.shape

    # Neural Turing Machine Layer
    memory = Memory(memory_shape, name='memory', memory_init=lasagne.init.Constant(1e-6), learn_init=False)
    controller = DenseController(l_input, memory_shape=memory_shape,
        num_units=num_units, num_reads=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        name='controller')
    heads = [
        WriteHead(controller, num_shifts=3, memory_shape=memory_shape, name='write', learn_init=False,
            nonlinearity_key=lasagne.nonlinearities.rectify,
            nonlinearity_add=lasagne.nonlinearities.rectify),
        ReadHead(controller, num_shifts=3, memory_shape=memory_shape, name='read', learn_init=False,
            nonlinearity_key=lasagne.nonlinearities.rectify)
    ]
    l_ntm = NTMLayer(l_input, memory=memory, controller=controller, heads=heads)

    # Output Layer
    l_output_reshape = ReshapeLayer(l_ntm, (-1, num_units))
    l_output_dense = DenseLayer(l_output_reshape, num_units=size + 1, nonlinearity=lasagne.nonlinearities.sigmoid, \
        name='dense')
    l_output = ReshapeLayer(l_output_dense, (batch_size, seqlen, size + 1))

    return l_output, l_ntm


if __name__ == '__main__':
    z=7
    b=1e-4
    for c in [150]:
        z=z+1
        # Define the input and expected output variable
        input_var, target_var = T.tensor3s('input', 'target')
        # The generator to sample examples from
        # generator = SumTask(batch_size=1, max_iter=1000000, size=2, max_length=5, end_marker=True)
        generator = SumTask(batch_size=1,max_iter=1000000, max_length=5, end_marker=True)
        # The model (1-layer Neural Turing Machine)
        l_output, l_ntm = model(input_var, \
            size=2, num_units=c, memory_shape=(128,20))
        # The generated output variable and the loss function
        pred_var = T.clip(lasagne.layers.get_output(l_output), 1e-6, 1. - 1e-6)
        loss = T.mean(lasagne.objectives.binary_crossentropy(pred_var, target_var))
        # Create the update expressions
        params = lasagne.layers.get_all_params(l_output, trainable=True)
        updates = graves_rmsprop(loss, params, learning_rate=b)
        # Compile the function for a training step, as well as the prediction function and
        # a utility function to get the inner details of the NTM
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        ntm_fn = theano.function([input_var], pred_var)
        ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, get_details=True))

        # Training
        try:
            scores, all_scores = [], []
            es=[]
            for i, (example_input, example_output) in generator:
                score = train_fn(example_input, example_output)
                scores.append(score)
                all_scores.append(round(score,5))
                if i % 500 == 0:
                    mean_scores = np.mean(scores)
                    print 'Batch #%d: %.6f' % (i, mean_scores)
                    scores = []
                    ls=[]
                    results=[]
                    for i, (example_input, example_output) in generator:
                        pred = ntm_fn(example_input)
                        results.append([example_input,example_output,pred])
                        if i%100==0:
                            for i,x in enumerate(results):
                                l = cross_entropy(np.array(x[2]), np.array(x[1]))
                                ls.append(round(l,5))
                            e=np.mean(np.array(ls))
                            es.append(e)
                            break
                if i>30000:
                    raise KeyboardInterrupt
                    break

        except KeyboardInterrupt:
            pass

        a={}
        a['scores']=all_scores
        a['terrors']=es
        a['learning_rate']=b
        a['units']=c
        x=json.dumps(a)
        with open('data_'+str(z)+'_'+str(b)+'_'+str(c)+'.json', 'w') as outfile:
            json.dump(x, outfile)
                


        # Visualization
        markers = [
            {
                'location': (lambda params: params['length']),
                'style': {'color': 'red'}
            }
        ]

        dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
            memory_shape=(128,20), markers=markers, cmap='bone')

        # Example
        params = generator.sample_params()
        dashboard.sample(**params)
        learning_curve(all_scores,'data_'+str(z)+'_'+str(b)+'_'+str(c))
