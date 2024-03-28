import numpy as np
import scipy.sparse as sp

"""
def shift(proj):
    shiftx = proj - np.max(proj, axis=0, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=0, keepdims=True)
        
def shiftY(proj):
    shiftx = proj - np.max(proj, axis=1, keepdims=True)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1, keepdims=True)

   
def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

"""


def he_initializer(n_inputs, n_outputs):
    """
    Initialization method that is specifically designed for activation functions that have an unbounded positive range, such as ReLU.
    """
    sd = np.sqrt(2 / n_inputs)
    return np.random.normal(loc=0, scale=sd, size=(n_inputs, n_outputs))


def glorot_initializer(n_inputs, n_outputs):
    """
    Glorot uniform initialization, also known as Xavier initialization.
    This weight initialization method aims to keep the variances of the input and output activations the same across layers, 
    facilitating the flow of gradients during training.
    
    In the original Glorot initialization paper, it was assumed that the weight matrix is used with a symmetric activation function 
    that has an output range of approximately (-1, 1). Examples of such activation functions are tanh and logistic sigmoid.
    This is also recommended for softmax activation function.
    """
    sd = np.sqrt(6.0 / (n_inputs + n_outputs))
    return np.random.uniform(-sd, sd, size=(n_inputs, n_outputs))


def binary_crossentropy(ytrue, ypred):
    """
    Binary crossentropy
    $ - \frac{1}{N} \sum_{i=1}^{N} ( y_i \log (p(y_i)) + (1 - y_i) \log (1 - p(y_i)) ) $ , where:
    $N$ is the number of nodes in the graph, 
    $y_i$ is the true label for the ith node 
    $p(y_i)$ is the predicted label for the ith node
    
    Arguments:
    `ytrue` is a numpy array contaigning true labels for all nodes
    `ypred` is a numpy array contaigning the predicted label values for all nodes
    """
    return -np.mean(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))


def preprocess_adjacency(A, symmetric):
    """
    Add self-loop to adjacency matrix and normalize the resulting sum.
    """
    assert A.shape[0] == A.shape[1], "Adjacency matrix must be squared."
    
    A_aux = A + np.eye(A.shape[0]) # add self-connections

    D_aux = np.zeros_like(A_aux)
    np.fill_diagonal(D_aux, np.asarray(A_aux.sum(axis=1)).flatten())

    if symmetric:
        D_aux_invroot = np.linalg.inv(sqrtm(D_aux))
        A_hat = D_aux_invroot @ A_aux @ D_aux_invroot
    else:
        D_aux_invroot = np.linalg.inv(D_aux)
        A_hat = D_mod_invroot @ A_mod
                
    return A_hat
        
        
def _preprocess_features(features): # EVALUAR SI ESCRIBIRLA DE CERO EN NUMPY
    """
    Copy from https://github.com/danielegrattarola/spektral/blob/39fe897c5c06ce8bd8100e10fe9d373b91958cc7/spektral/datasets/citation.py#L192
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class GradDescentOptim(): # EVALUAR SI DEJAR ESTE OPTIMIZADOR O IMPLEMENTAR ADAM
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None
        
    def __call__(self, y_pred, y_true, train_nodes=None):
        self.y_pred = y_pred
        self.y_true = y_true
        
        if train_nodes is None:
            self.train_nodes = np.arange(y_pred.shape[0])
        else:
            self.train_nodes = train_nodes
            
        self.bs = self.train_nodes.shape[0]
        
    @property
    def out(self):
        return self._out
    
    @out.setter
    def out(self, y):
        self._out = y
    

out_degrees = np.sum(A_mod, axis=1)
inv_degrees = 1.0 / out_degrees
D_mod_inv = np.diag(inv_degrees)

    

class GCNLayer(): # EVALUAR SI QUITAR EL ARGUMENTO "ACTIVATION" O JUNTAR GCN Y SOFTMAX
    """
    Important:
        GCNLayer expects A_hat, which is the result of adding self-loops to the adjacency matrix before normalization.
        A_hat does not change in the network, so the idea is to calculate it before starting, instead of calculating it at each layer.
        This class does not check if A_hat is the matrix given to the `forward` method. 
    """
    def __init__(self, n_inputs, n_outputs, activation, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_initializer(self.n_outputs, self.n_inputs) # Nota: en caso de usar ReLU como activación, es conveniente usar el inicializador de pesos He
        self.b = np.zeros((self.n_outputs, 1))
        self.activation = activation
        self.name = name
        
    def __repr__(self):
        return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs}) {self.activation}"
           
    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)
            
    def forward(self, A_hat, X, W=None):
        """
        Assumes A is (bs, bs) adjacency matrix and X is (bs, D), 
            where bs = "batch size" and D = input feature length
        """
        
        self._X = (A_hat @ X).T # for calculating gradients.  (D, bs)
        
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        
        H = (W @ self._X) + b # (h, D)*(D, bs) -> (h, bs)
        if self.activation is "tanh":
            H = np.tanh(H)
        if self.activation is "softmax":
            H = self.shift(H)
        else:
            raise ValueError("activation must be either \"tanh\" or \"softmax\"")
        self._H = H # (h, bs)
        return self._H.T # (bs, h)

    
    def backward(self, optim, update=True):
        dtanh = 1 - np.asarray(self._H.T)**2 # (bs, out_dim)
        d2 = np.multiply(optim.out, dtanh)  # (bs, out_dim) *element_wise* (bs, out_dim)
        
        self.grad = self._A @ d2 @ self.W # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)     
        optim.out = self.grad
        
        dW = np.asarray(d2.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:
            self.W -= (dW + dW_wd) * optim.lr 
        
        return dW + dW_wd
        
        # should take in optimizer, update its own parameters and update the optimizer's "out"
        # Build mask on loss
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))
        
        # derivative of loss w.r.t. activation (pre-softmax)
        d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out_dim)
        d1 = np.multiply(d1, train_mask) # (bs, out_dim) with loss of non-train nodes set to zero
        
        self.grad = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        optim.out = self.grad
        
        dW = (d1.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)
                
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:   
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)


    
class SoftmaxLayer():
    def __init__(self, n_inputs, n_outputs, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.b = np.zeros((self.n_outputs, 1))
        self.name = name
        self._X = None # Used to calculate gradients
        
    def __repr__(self):
        return f"Softmax: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
    
    def shift(self, proj):
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)
        
    def forward(self, X, W=None, b=None):
        """Compute the softmax of vector x in a numerically stable way.
        
        X is assumed to be (bs, h)
        """
        self._X = X.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        proj = np.asarray(W @ self._X) + b # (out, h)*(h, bs) = (out, bs)
        return self.shift(proj).T # (bs, out)
    
    def backward(self, optim, update=True):
        # should take in optimizer, update its own parameters and update the optimizer's "out"
        # Build mask on loss
        train_mask = np.zeros(optim.y_pred.shape[0])
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))
        
        # derivative of loss w.r.t. activation (pre-softmax)
        d1 = np.asarray((optim.y_pred - optim.y_true)) # (bs, out_dim)
        d1 = np.multiply(d1, train_mask) # (bs, out_dim) with loss of non-train nodes set to zero
        
        self.grad = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        optim.out = self.grad
        
        dW = (d1.T @ self._X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)
                
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:   
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)


gcn1 = GCNLayer(g.number_of_nodes(), 2, activation=np.tanh, name='1')
sm1 = SoftmaxLayer(2, n_classes, "SM")
opt = GradDescentOptim(lr=0, wd=1.)

gcn1_out = gcn1.forward(A_hat, X)
opt(sm1.forward(gcn1_out), labels)

