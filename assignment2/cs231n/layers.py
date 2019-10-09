from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    _x = np.reshape(x, (x.shape[0], -1))
    out = _x.dot(w)
    out = out + b.reshape(1, -1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)
    dx = np.reshape(dx, x.shape)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0.0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (x > 0) * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean = x.mean(axis=0)
        var = 1 / N * np.sum((x - mean) ** 2, axis=0)
        _x = (x - mean) / np.sqrt(var + eps)
        out = gamma * _x + beta
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x, _x, gamma, beta, mean, var, eps)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = dout.shape
    x, _x, gamma, beta, mean, var, eps= cache
    dgamma = np.sum(dout * _x, axis=0)
    dbeta = np.sum(dout, axis=0)
    d_x = gamma * np.ones(_x.shape) * dout
    dvar = -np.sum(1 / (2 * np.sqrt(var + eps) * (var + eps)) * d_x * (x - mean), axis=0)
    dmean = -np.sum(d_x / np.sqrt(var + eps), axis=0)
    dmean += 2 / N * np.sum(mean - x, axis=0) * dvar
    dx = d_x / np.sqrt(var + eps)
    dx += 2 / N * (x - mean) * dvar
    dx += 1 / N * dmean
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = dout.shape
    x, _x, gamma, beta, mean, var, eps= cache
    dgamma = np.sum(dout * _x, axis=0)
    dbeta = np.sum(dout, axis=0)
    d_x = gamma * np.ones(_x.shape) * dout
    dvar = -0.5 * np.sum(d_x * _x , axis=0) / (var + eps)
    dmean = -np.sum(d_x / np.sqrt(var + eps), axis=0) - 2 / N * np.sum(_x , axis=0) * np.sqrt(var + eps) * dvar    
    dx = d_x / np.sqrt(var + eps)
    dx += 2 / N * (x - mean) * dvar
    dx += 1 / N * dmean
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    '''
    N, D = x.shape
    mean = x.mean(axis=1, keepdims=True)
    var = 1 / D * np.sum((x - mean) ** 2, axis=1, keepdims=True)
    _x = (x - mean) / np.sqrt(var + eps)
    out = gamma * _x + beta
    cache = x, _x, gamma, beta, mean, var, eps 
    '''
    x = x.T
    N, D = x.shape
    mean = x.mean(axis=0)
    var = 1 / N * np.sum((x - mean) ** 2, axis=0)
    _x = (x - mean) / np.sqrt(var + eps)
    out = gamma * _x.T + beta
    cache = (x.T, _x.T, gamma, beta, mean, var, eps)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''   
    dout = dout.T
    N, D = dout.shape
    x, _x, gamma, beta, mean, var, eps= cache
    dgamma = np.sum(dout.T * _x, axis=0)
    dbeta = np.sum(dout.T, axis=0)
    d_x = gamma * np.ones(_x.shape) * dout.T

    dvar = -0.5 * np.sum(d_x * _x , axis=0) / (var + eps)
    dmean = -np.sum(d_x / np.sqrt(var + eps), axis=0) - 2 / N * np.sum(_x , axis=0) * np.sqrt(var + eps) * dvar    
    dx = d_x / np.sqrt(var + eps)
    dx += 2 / N * (x - mean) * dvar
    dx += 1 / N * dmean
    '''
    dout = dout.T
    N, D = dout.shape
    x, _x, gamma, beta, mean, var, eps= cache
    x = x.T
    _x = _x.T
    dgamma = np.sum(dout * _x, axis=1)
    dbeta = np.sum(dout, axis=1)
    d_x = gamma * np.ones(_x.T.shape) * dout.T
    sum_x = np.sum(_x , axis=0)
    dvar = -0.5 * np.sum(d_x * _x.T , axis=1) / (var + eps)
    dmean = -np.sum(d_x.T / np.sqrt(var + eps), axis=0) - 2 / N * np.sum(_x , axis=0) * np.sqrt(var + eps) * dvar    
    dx = d_x.T / np.sqrt(var + eps)
    dx += 2 / N * (x - mean) * dvar
    dx += 1 / N * dmean
    dx = dx.T


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.random(x.shape)
        mask = (mask > 1 - p) * 1
        out = x * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout / dropout_param['p']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    k = w
    h, w = 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride

    # pad_x = np.pad(x, (pad, pad), mode='constant')
    # print(pad_x.shape)
    out = np.zeros((N, F, h, w))
    def conv(x, k):
        x = np.pad(x, (pad, pad), mode='constant')
        res = np.zeros((h, w))
        begin_x = []
        begin_y = []
        begin = 0
        while begin + HH <= H + 2 * pad:
            begin_x.append(begin)
            begin += stride
        begin = 0
        assert len(begin_x) == h
        while begin + WW <= W + 2 * pad:
            begin_y.append(begin)
            begin += stride
        assert len(begin_y) == w
        for i_x, beginx in enumerate(begin_x):
            for j_y, beginy in enumerate(begin_y):
                for i in range(HH):
                    for j in range(WW):
                        res[i_x, j_y] += x[i + beginx, j + beginy] * k[i, j]
        return res


        
    for i in range(N):
        for j in range(F):
            for c in range(C):
                out[i, j] = out[i, j] + conv(x[i, c], k[j, c])
        out[i] += b.reshape((-1, 1, 1))
    # print(out)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, k, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache 
    db = dout.sum(axis=0).sum(axis=1).sum(axis=1)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    k = w
    h, w = 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride
    def dconv(dout, x, k):
        x = np.pad(x, (pad, pad), mode='constant')
        res = np.zeros((h, w))
        dx = np.zeros_like(x)
        dk = np.zeros_like(k)
        begin_x = []
        begin_y = []
        begin = 0
        while begin + HH <= H + 2 * pad:
            begin_x.append(begin)
            begin += stride
        begin = 0
        assert len(begin_x) == h
        while begin + WW <= W + 2 * pad:
            begin_y.append(begin)
            begin += stride
        assert len(begin_y) == w
        for i_x, beginx in enumerate(begin_x):
            for j_y, beginy in enumerate(begin_y):
                for i in range(HH):
                    for j in range(WW):
                        #res[i_x, j_y] += x[i + beginx, j + beginy] * k[i, j]
                        dx[i + beginx, j + beginy] += dout[i_x, j_y] * k[i, j]
                        dk[i, j] += dout[i_x, j_y] * x[i + beginx, j + beginy]
        return dx[pad:-pad if pad != 0 else None, pad:-pad if pad != 0 else None], dk
    for i in range(N):
        for j in range(F):
            for c in range(C):
                #out[i, j] = out[i, j] + conv(x[i, c], k[j, c])
                a, b = dconv(dout[i, j], x[i, c], k[j, c])
                dx[i, c] += a
                dw[j, c] += b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache = (x, pool_param)
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    h = 1 + (H - pool_height) // stride
    w = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, h, w))

    def max_pool(x):
        begin = 0
        begin_x = []
        begin_y = []
        res = np.zeros((h, w))
        while begin + pool_height <= H:
            begin_x.append(begin)
            begin += stride
        begin = 0
        while begin + pool_width <= W:
            begin_y.append(begin)
            begin += stride
        for i_x, beginx in enumerate(begin_x):
            for i_y, beginy in enumerate(begin_y):
                res[i_x, i_y] = x[beginx:beginx + pool_height, beginy:beginy + pool_width].max()
        return res
    for i in range(N):
        for j in range(C):
            out[i, j] = max_pool(x[i, j])
    #print(out)




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    h = 1 + (H - pool_height) // stride
    w = 1 + (W - pool_width) // stride  
    dx = np.zeros_like(x)

    def dmax_pool(dout, x):
        begin = 0
        begin_x = []
        begin_y = []
        dx = np.zeros_like(x)
        while begin + pool_height <= H:
            begin_x.append(begin)
            begin += stride
        begin = 0
        while begin + pool_width <= W:
            begin_y.append(begin)
            begin += stride
        for i_x, beginx in enumerate(begin_x):
            for i_y, beginy in enumerate(begin_y):
                index = np.unravel_index(x[beginx:beginx+pool_height, beginy:beginy+pool_width].argmax(), x[beginx:beginx+pool_height, beginy:beginy+pool_width].shape)
                index_d = (index[0] + beginx, index[1] + beginy)
                dx[index_d] = dout[i_x, i_y]
                # max_x = -1e9
                # max_i = 0
                # max_j = 0
                # for i in range(beginx, beginx + pool_height):
                #     for j in range(beginy, beginy + pool_width):
                #         if x[i, j] > max_x:
                #             max_x = x[i, j]
                #             max_i = i
                #             max_j = j
                # dx[max_i, max_j] = dout[i_x, i_y]
        return dx

    for i in range(N):
        for j in range(C):
            #out[i, j] = max_pool(x[i, j])
            dx[i, j] = dmax_pool(dout[i, j], x[i, j])
            
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = 1e-6
    momentum = bn_param['momentum'] if 'momentum' in bn_param else 0.9
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))
    if mode == 'train':
        mean = x.mean(axis=0).mean(axis=1).mean(axis=1)
        var = 1.0 / (N * H * W) * np.sum((x - mean.reshape(1, -1, 1, 1)) ** 2, axis=0).sum(axis=1).sum(axis=1)
        running_mean = running_mean * momentum + (1.0 - momentum) * mean
        running_var = running_var * momentum + (1.0 - momentum) * var
        _x = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + eps)
        cache = x, _x, gamma, beta, mean, var, eps
    elif mode == 'test':
        _x = (x - running_mean.reshape(1, -1, 1, 1)) / np.sqrt(running_var.reshape(1, -1, 1, 1) + eps)
    out = _x * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    x, _x, gamma, beta, mean, var, eps= cache
    dgamma = np.sum(dout * _x, axis=0).sum(axis=1).sum(axis=1)
    dbeta = np.sum(dout, axis=0).sum(axis=1).sum(axis=1)
    d_x = gamma.reshape(1, -1, 1, 1) * np.ones(_x.shape) * dout
    dvar = -0.5 * np.sum(d_x * _x , axis=0).sum(axis=1).sum(axis=1) / (var + eps)
    dmean = -np.sum(d_x / np.sqrt(var + eps).reshape(1, -1, 1, 1), axis=0).sum(axis=1).sum(axis=1) - 2 / (N * H * W) * np.sum(_x , axis=0).sum(axis=1).sum(axis=1) * np.sqrt(var + eps) * dvar    
    dx = d_x / np.sqrt(var.reshape(1, -1, 1, 1) + eps)
    dx += 2 / (N * H * W) * (x - mean.reshape(1, -1, 1, 1)) * dvar.reshape(1, -1, 1, 1)
    dx += 1 / (N * H * W) * dmean.reshape(1, -1, 1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    y = x
    eps = gn_param['eps']
    gx = np.split(x, G, axis=1)
    outs = []
    means = []
    varis = []
    for x in gx: 
        mean = x.mean(axis=1).mean(axis=1).mean(axis=1)
        var = 1.0 / (C / G * H * W) * np.sum((x - mean.reshape(-1, 1, 1, 1)) ** 2, axis=1).sum(axis=1).sum(axis=1)
        _x = (x - mean.reshape(-1, 1, 1, 1)) / np.sqrt(var.reshape(-1, 1, 1, 1) + eps)
        outs.append(_x)
        means.append(mean)
        varis.append(var)
    gx = np.concatenate(outs, axis=1)
    #assert x.shape == gx.shape
    out = gx * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)
    cache = y, outs, gamma, beta, means, varis, eps, G

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    x, _x, gamma, beta, mean, var, eps, G = cache
    dout = np.split(dout, G, axis=1)
    dgamma = []
    dbeta = []
    for i, __x in enumerate(_x):
        dgamma.append(np.sum(dout[i] * __x, axis=0).sum(axis=1).sum(axis=1))
        dbeta.append(np.sum(dout[i], axis=0).sum(axis=1).sum(axis=1))

    dx = x
    dgamma = np.concatenate(dgamma, axis=0)
    dbeta = np.concatenate(dbeta, axis=0)
    # d_x = gamma.reshape(1, -1, 1, 1) * np.ones(_x.shape) * dout
    # dvar = -0.5 * np.sum(d_x * _x , axis=0).sum(axis=1).sum(axis=1) / (var + eps)
    # dmean = -np.sum(d_x / np.sqrt(var + eps).reshape(1, -1, 1, 1), axis=0).sum(axis=1).sum(axis=1) - 2 / (N * H * W) * np.sum(_x , axis=0).sum(axis=1).sum(axis=1) * np.sqrt(var + eps) * dvar    
    # dx = d_x / np.sqrt(var.reshape(1, -1, 1, 1) + eps)
    # dx += 2 / (N * H * W) * (x - mean.reshape(1, -1, 1, 1)) * dvar.reshape(1, -1, 1, 1)
    # dx += 1 / (N * H * W) * dmean.reshape(1, -1, 1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
