"""Local approximation technique.

References:

1. Farmer JD, Sidorowich JJ. Predicting chaotic time series. Phys Rev Lett. (1987)

2. Packard, N.H., Crutchfield, J.P., Farmer, J.D. & Shaw, R.S. (1980). 
'Geometry from a Time Series'. Physical Review Letters, 45(9), pp.712-716.

3.

"""

# Author: Starkov Artyom
# License: BSD3
import warnings

import numpy as np
from sklearn.linear_model import Ridge

from scipy import interpolate
from sklearn.neighbors import NearestNeighbors

from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    RegressorMixin
    )

from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y
    )

class PackardTransformer(TransformerMixin, BaseEstimator):
    """ 
    Transform array to new sample of delayed initial array.

    
    """
    
    def __init__(self, dim=2, delay=1):
        self.dim = dim
        self.delay = delay   
    
    def _delays(self, X):

        min_size = X.size - self.delay*(self.dim - 1)
        space = np.ndarray([self.dim, min_size])
        
        for i in range(self.dim):
            start = i*self.delay
            space[i, :] = X[start:start + min_size]
        return space
    
    def _check_params(self, X):
        
        if self.dim > self.n_features_:
            raise ValueError('Parameter `dim` must be less or equal n_features.')
        
        elif X.size - self.delay*(self.dim - 1) <= 0:
            raise ValueError('Not available combination of parameters. '
                             'Equation: n_features - delay*(dim - 1) must be more 0.')
        else:
            return X

    def fit(self, X, y=None):
        X = check_array(X, ensure_min_features=2)
        self.n_features_ = X.shape[1]
        X = self._check_params(X)
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, ensure_min_features=2)

        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`.')
        
        return np.apply_along_axis(self._delays, 1, X)



class LocalStates(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.
    
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
        
    Examples
    --------
    >> X = np.random.random((3, 5))
    >> X
    
              --- t1 ---   --- t2 ---  --- t3 --- --- t4 ---   --- t5 ---
       array([[0.43364583, 0.5600714 , 0.17840582, 0.61125872, 0.45059375], --- X
             [0.10804393, 0.23509252, 0.66908984, 0.46054161, 0.03609657],  --- Y
             [0.61973627, 0.22499159, 0.65306052, 0.58212125, 0.31095344]]) --- Z
    
    >> st = LocalStates(n_neighbors=2, n_next=2).fit_transform(X)
    >> st
    
                --- tn --- -- (tn + 1) -- -- (tn + 2) --
       array([[[0.17840582, 0.61125872, 0.45059375],     --- n1 ---   X
               [0.43364583, 0.5600714 , 0.17840582]],    --- n2 ---

               [[0.66908984, 0.46054161, 0.03609657],    --- n1 ---   Y
                [0.10804393, 0.23509252, 0.66908984]],   --- n2 ---

               [[0.65306052, 0.58212125, 0.31095344],    --- n1 ---   Z
                [0.61973627, 0.22499159, 0.65306052]]])  --- n2 ---
        
        where, 
                ni - nearest neighbor
                tn - index of future states
        
    >> np.stack(st, axis=1)
    
       array([[[0.17840582, 0.61125872, 0.45059375],  --- X 
               [0.66908984, 0.46054161, 0.03609657],  --- Y ---  n1
               [0.65306052, 0.58212125, 0.31095344]], --- Z
              
              [[0.43364583, 0.5600714 , 0.17840582],  --- X
               [0.10804393, 0.23509252, 0.66908984],  --- Y ---  n2
               [0.61973627, 0.22499159, 0.65306052]]])--- Z
               
    """
    def __init__(self, n_neighbors=1, n_query=1, n_next=0,
                 how='point', tree='kd_tree', interp=None):
        self.n_neighbors = n_neighbors
        self.n_query = n_query
        self.n_next = n_next
        self.how = how
        self.tree = tree
        self.interp = interp
    
    def _how(self, how):
        if how == 'point':
            return self._point
        elif how == 'line':
            return self._line
        else:
            raise ValueError('Please change'
                             '`how` parameter to `point` or `line`')
            
    def _point(self, x, start, n_neighbr, n_query, tree_type):
        return self._method(x, start, n_neighbr, n_query, tree_type)
    
    def _traj(self, ids, k):
        start = 0
        
        for i in range(len(ids)):
            if i == len(ids)-1:
                yield ids[start:i + 1]

            elif abs(ids[i+1] - ids[i]) > 1:
                yield ids[start:i + 1]
                start = i + 1

    def _line(self, x, start, n_neighbr, n_query, tree_type):
        dist, ids = self._method(x, start, n_neighbr, n_query, tree_type)

        ids.sort()
                
        lines = [i for i in self._traj(ids.reshape(-1, 1), 1)]
                
        nbrs_id = []
        nbrs_dist = []
                
        for line in lines:
            min_dist = min(dist[0][line - line[0]])
            min_id  = np.where(dist == min_dist) + line[0]
                    
            nbrs_dist.append(min_dist)
            nbrs_id.append(min_id[0][0])
        
        return np.array(nbrs_dist), np.array(nbrs_id)
    
    def _method(self, x, start, n_neighbr, n_query, tree_type):
        
        if tree_type == 'kd_tree' or tree_type == 'ball_tree':
            nn = NearestNeighbors(n_neighbors=n_neighbr, algorithm=tree_type)
            nn.fit(x)
            
            knn = nn.kneighbors(start)
        else:
            raise ValueError('Known tree are: `kd_tree`, `ball_tree`, `lsh`')
        
        return knn
    
    def _interpolate(self, X, n):
        """
        """
        
        def inter(X, n):
            length = X.size
            ind = np.linspace(X.min(), X.max(), length)
            new_ind = np.linspace(X.min(), X.max(), length * n)
            return interpolate.interp1d(ind, X, kind='linear')(new_ind)
        
        return np.apply_along_axis(inter, 1, X, n)
    
    def _check_params(self, X):
        if self.n_next >= self.n_features_:
            raise ValueError('Parameter `n_next` must be less than dataset number of features.')
                
        elif self.n_neighbors > self.n_features_ - self.n_next:
            raise ValueError('Sum of n_neighbors and n_next must be less or equal n_features.') 
                    
        elif self.n_query < 1:
            raise ValueError('Needed at least 1 point for search.')
                
        elif self.n_query > self.n_features_:
            raise ValueError(f'Parameter `n_query` must be less or equal n_features.')
        
        else:
            return X
    
    def get_distance(self):
        return self._dists
        
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
            
        """
        X = check_array(X, ensure_min_features=2)
        self.n_features_ = X.shape[1]
        X = self._check_params(X)

        method = self._how(self.how)
            
        if self.interp is None:
            n_next = self.n_next
            step = 1
        else:
            X = self._interpolate(X, self.interp)
            self._interp_X = X
            step = self.interp
            n_next = step*self.n_next
            
        XT = X.T
        start = XT[-self.n_query:]
            
        if self.n_next != 0:
            XT = XT[:-n_next]
            
        self._dists, idxs = method(XT, start, self.n_neighbors, self.n_query, self.tree)
        
        idxs = idxs.reshape(-1, 1)
        n_next = np.arange(0, n_next + 1, step).reshape(1, -1)
            
        next_idxs = np.repeat(n_next, 
                              idxs.size,
                              axis=0)
        
        self._full_idxs = idxs + next_idxs

        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
            
       
        """
        
        check_is_fitted(self, 'n_features_')
        X = check_array(X, ensure_min_features=2)

        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        if self.interp is not None:
            X = self._interp_X
        local = [X[:, idx] for idx in self._full_idxs]
        return np.stack(local, axis=1)


class LocalModel(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator(demo_param='demo_param')
    """
    def __init__(self, dim=1, delay=1, n_neighbors=1, 
                 n_query=1, n_next=1, how='point', 
                 tree='kd_tree', interp=None, reg=1, target='multi'):
        self.dim = dim
        self.delay = delay
        self.n_neighbors = n_neighbors
        self.n_query = n_query
        self.n_next = n_next
        self.how = how
        self.tree = tree
        self.interp = interp
        self.reg = reg
        self.target = target
    
    def _approximate(self, X, y):
        """
        
        """
        
        if self.reg is not None:
            regressor = Ridge(alpha=self.reg, solver='auto', normalize=True, tol=1e-10)
        else:
            pass
        
        if self.target == 'multi':
            targets = MultiOutputRegressor(regressor).fit(X, y)
        elif self.target == 'variate':
            targets = regressor.fit(X, y)
        elif self.target == 'chain':
            targets = RegressorChain(regressor).fit(X, y)
        elif self.target == 'mlpr':
            from sklearn.neural_network import MLPRegressor
            targets = MLPRegressor(max_iter=1500,hidden_layer_sizes=3, alpha=0.1, learning_rate_init=0.000001).fit(X, y)
        else:
            raise ValueError('')
        
        return targets

    def _get_X_y(self, spaces):
        
        ls = LocalStates(self.n_neighbors, self.n_query, self.n_next, 
                        self.how, self.tree, self.interp)

        for space in spaces:
            states = ls.fit_transform(space)
            
            stacked = np.stack(states, axis=1)
            nn_states = stacked.T[0]
            
            y = states[-1][:, 1:]
            X = nn_states.T
            
            yield X, y

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, ensure_min_features=2)
        self.is_fitted_ = True
        
        self._models = []
        self._spaces = PackardTransformer(self.dim, self.delay).fit_transform(X)

        for X, y in self._get_X_y(self._spaces):
            self._y = y
            self._models.append(self._approximate(X, y))
        return self
    
    def predict(self, X, y=None):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred = []
        
        for i, model in enumerate(self._models):
            pred.append(model.predict(self._spaces[i][:, -1].reshape(1, -1))[0])
        return np.array(pred)