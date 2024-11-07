from operator import itemgetter
from functools import reduce,partial
import pandas as pd
import autograd.numpy as np

from seldonian.models.objectives import (
    sample_from_statistic,evaluate_statistic)
from seldonian.utils.stats_utils import *

class Node(object):
    def __init__(self,name,lower,upper):
        """The base class for all parse tree nodes
        
        :param name: 
            The name of the node
        :type name: str

        :param lower: 
            Lower confidence bound
        :type lower: float

        :param upper: 
            Upper confidence bound
        :type upper: float
        
        :ivar index: 
            The index of the node in the tree
        :vartype index: int

        :ivar left: 
            Left child node
        :vartype left: Node object, defaults to None

        :ivar right: 
            Right child node
        :vartype right: Node object, defaults to None
        
        :ivar will_lower_bound: 
            Whether this node needs a lower bound
        :vartype will_lower_bound: bool
        
        :ivar will_upper_bound: 
            Whether this node needs an upper bound
        :vartype will_upper_bound: bool

        """
        self.name = name
        self.index = None 
        self.left  = None 
        self.right = None 
        self.lower = lower 
        self.upper = upper 
        self.will_lower_bound = True
        self.will_upper_bound = True

    def __repr__(self):
        """ The string representation of the node. 
        Also, what is displayed inside the node box 
        in the visual graph 
        """
        lower_bracket = '(' if np.isinf(self.lower) else '[' 
        upper_bracket = ')' if np.isinf(self.upper) else ']'

        lower_str = f'{self.lower:g}' if self.will_lower_bound else '_'
        upper_str = f'{self.upper:g}' if self.will_upper_bound else '_'


        bounds_str = \
            f'{lower_bracket}{lower_str}, {upper_str}{upper_bracket}' \
            if (self.lower!= None or self.upper!=None) else '()'

        return '\n'.join(
            [
                '['+str(self.index)+']',
                str(self.name),
                u'\u03B5' + ' ' + bounds_str
            ]
        ) 
  

class BaseNode(Node):
    def __init__(self,
        name,
        lower=float('-inf'),
        upper=float('inf'),
        conditional_columns=[],
        **kwargs):
        """ Class for base variable leaf nodes
        in the parse tree.

        :param name: 
            The name of the node
        :type name: str
        :param lower: 
            Lower confidence bound
        :type lower: float
        :param upper: 
            Upper confidence bound
        :type upper: float
        :param conditional_columns: 
            When calculating confidence bounds on a measure 
            function, condition on these columns being == 1
        :type conditional_columns: List(str)
        :ivar node_type: 
            equal to 'base_node'
        :vartype node_type: str
        :ivar delta: 
            The share of the confidence put into this node
        :vartype delta: float
        :ivar measure_function_name: str
            The name of the statistical measurement
            function that this node represents, e.g. "FPR". 
            Must be contained in measure_functions
            list in :py:mod:`.operators`
        :vartype measure_function_name: str
        """
        super().__init__(name,lower,upper,**kwargs)
        self.conditional_columns = conditional_columns
        self.node_type = 'base_node'
        self.delta = 0  
        self.measure_function_name = '' 

    def __repr__(self):
        """ Overrides Node.__repr__()
        """
        return super().__repr__() + ', ' + u'\u03B4' + f'={self.delta:g}'
    
    def calculate_value(self,
        **kwargs):
        """
        Calculate the value of the node 
        given model weights, etc. This is
        the expected value of the base variable,
        not the bound.
        """ 
        value = evaluate_statistic(
            statistic_name=self.measure_function_name,
            **kwargs)
        return value

    def mask_data(self,
        dataset,
        conditional_columns):
        """Mask features and labels using 
        a joint AND mask where each of the
        conditional columns is True.

        :param dataset: 
            The candidate or safety dataset
        :type dataset: dataset.Dataset object
        :param conditional_columns: 
            List of columns for which to create
            the joint AND mask on the dataset
        :type conditional_columns: List(str)

        :return: The masked dataframe 
        :rtype: numpy ndarray
        """
        # Figure out indices of sensitive attributes from their column names
        sensitive_col_indices = [dataset.sensitive_col_names.index(
            col) for col in conditional_columns]

        joint_mask = reduce(np.logical_and,
            (dataset.sensitive_attrs[:,col_index]==1 for col_index in sensitive_col_indices))
        if dataset.regime == 'supervised_learning':
            if type(dataset.features) == list:
                masked_features = [x[joint_mask] for x in dataset.features]
                # If possible, convert to numpy array. Not always possible, 
                # e.g., if features are of different dimensions.
                try:
                    masked_features = np.array(masked_features)
                    n_masked = len(masked_features)
                except Exception as e:
                    # masked_features and masked_labels stay as lists
                    n_masked = len(masked_features[0])
            else:
                # numpy array 
                masked_features = dataset.features[joint_mask]
                n_masked = len(masked_features)
            print("dataset.features", dataset.features.shape)
            print("dataset.labels", dataset.labels[0].shape)
            print("dataset.sensitive_attrs", dataset.sensitive_attrs.shape)
            if type(dataset.labels) == list:
                masked_labels = [x[joint_mask] for x in dataset.labels]
                masked_labels = np.array(masked_labels)
            else:
                masked_labels = dataset.labels[joint_mask]
            return masked_features,masked_labels,n_masked
        
        elif dataset.regime == 'reinforcement_learning':
            masked_episodes = np.asarray(dataset.episodes)[joint_mask]
            n_masked = len(masked_episodes)
            return masked_episodes,n_masked

    def calculate_data_forbound(self,**kwargs):
        """
        Prepare data inputs
        for confidence bound calculation.
        """
        theta,dataset,model,regime,branch = itemgetter(
                    'theta','dataset','model',
                    'regime','branch')(kwargs)

        if branch == 'candidate_selection':
            # Then we're in candidate selection
            n_safety = kwargs['n_safety']

        # If in candidate selection want to use safety data size
        # in bound calculation
        
        if regime == 'supervised_learning':
            # mask the data using the conditional columns, if present
            
            features = dataset.features
            labels = dataset.labels
            if self.conditional_columns:
                masked_features,masked_labels,n_masked = self.mask_data(
                    dataset,self.conditional_columns)
            else:
                (masked_features,
                    masked_labels,
                    n_masked) = features,labels,dataset.num_datapoints

            if branch == 'candidate_selection':
                frac_masked = n_masked/dataset.num_datapoints
                datasize = int(round(frac_masked*n_safety))
            else:
                datasize = n_masked
            data_dict = {'features':masked_features,'labels':masked_labels}  
            
        elif regime == 'reinforcement_learning':
            gamma = model.env_kwargs['gamma']
            episodes = dataset.episodes

            if self.conditional_columns:
                masked_episodes,n_masked = self.mask_data(
                    dataset,self.conditional_columns)
            else:
                (masked_episodes,
                    n_masked) = episodes,dataset.num_datapoints

            if branch == 'candidate_selection':
                frac_masked = n_masked/dataset.num_datapoints
                datasize = int(round(frac_masked*n_safety))
            else:
                datasize = n_masked
            
            # Precalculate expected return from behavioral policy
            masked_returns = [weighted_sum_gamma(ep.rewards,gamma) for ep in masked_episodes]

            data_dict = {
                'episodes':masked_episodes,
                'weighted_returns':masked_returns
            }

        return data_dict,datasize
                    
    def calculate_bounds(self,
        **kwargs):
        """Calculate confidence bounds given a bound_method, 
        such as t-test.
        """ 
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'manual':
                # Bounds set by user
                return {'lower':self.lower,
                        'upper':self.upper}

            elif bound_method == 'random':
                # Randomly assign lower and upper bounds
                lower, upper = (
                    np.random.randint(0,2),
                    np.random.randint(2,4)
                    )
                return {'lower':lower,'upper':upper}
        
            else:
                # Real confidence bound 

                # --TODO-- abstract away to support things like 
                # getting confidence intervals from bootstrap
                # and RL cases
                estimator_samples = self.zhat(**kwargs)

                branch = kwargs['branch']
                data_dict = kwargs['data_dict']
                bound_kwargs = kwargs
                bound_kwargs['data'] = estimator_samples
                bound_kwargs['delta'] = self.delta
                
                # If lower and upper are both needed, 
                # can't necessarily call lower and upper
                # bound functions separately. Sometimes the joint bound
                # is different from the individual bounds combined
                if self.will_lower_bound and self.will_upper_bound:
                    if branch == 'candidate_selection':
                        lower,upper = self.predict_HC_upper_and_lowerbound(
                            **bound_kwargs)
                    elif branch == 'safety_test':
                        lower,upper = self.compute_HC_upper_and_lowerbound(
                            **bound_kwargs)
                    return {'lower':lower,'upper':upper}
                
                elif self.will_lower_bound:
                    if branch == 'candidate_selection':
                        lower = self.predict_HC_lowerbound(
                            **bound_kwargs)
                    elif branch == 'safety_test':
                        lower = self.compute_HC_lowerbound(
                            **bound_kwargs)
                    return {'lower':lower}

                elif self.will_upper_bound:
                    if branch == 'candidate_selection':
                        upper = self.predict_HC_upperbound(
                            **bound_kwargs)
                    elif branch == 'safety_test':
                        upper = self.compute_HC_upperbound(
                            **bound_kwargs)
                    return {'upper':upper}

                raise AssertionError(
                    "will_lower_bound and will_upper_bound "
                    "cannot both be False") 

        else:
            raise RuntimeError("bound_method not specified!")
    
    def zhat(self,model,theta,data_dict,datasize,**kwargs):
        """
        Calculate an unbiased estimate of the 
        base variable node.
    
        :param model: The machine learning model
        :type model: models.SeldonianModel object
        :param theta: 
            model weights
        :type theta: numpy ndarray
        :param data_dict: 
            Contains inputs to model, 
            such as features and labels
        :type data_dict: dict
        """

        return sample_from_statistic(model=model,
            statistic_name=self.measure_function_name,
            theta=theta,data_dict=data_dict,
            datasize=datasize,**kwargs)

    def predict_HC_lowerbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence lower bound
        that we expect to pass the safety test.
        Used in candidate selection

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray 
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """ 
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']

            if bound_method == 'ttest':
                lower = data.mean() - 2.0*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
            else:
                raise NotImplementedError(f"Bounding method {bound_method} is not supported")
        
        return lower

    def predict_HC_upperbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence upper bound
        that we expect to pass the safety test.
        Used in candidate selection

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray 
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """  
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'ttest':
                lower = data.mean() + 2.0*stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
            else:
                raise NotImplementedError(f"Bounding method {bound_method} is not supported")
            
        return lower

    def predict_HC_upper_and_lowerbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence lower and upper bounds
        that we expect to pass the safety test.
        Used in candidate selection.
    
        Depending on the bound_method,
        this is not always equivalent
        to calling predict_HC_lowerbound() and 
        predict_HC_upperbound() independently.

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray 
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """ 
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'ttest':
                lower = self.predict_HC_lowerbound(data=data,
                    datasize=datasize,delta=delta/2,
                    **kwargs)
                upper = self.predict_HC_upperbound(data=data,
                    datasize=datasize,delta=delta/2,
                    **kwargs)

            elif bound_method == 'manual':
                pass
            else:
                raise NotImplementedError(
                    f"Bounding method {bound_method}"
                    " is not supported")

            
        return lower,upper

    def compute_HC_lowerbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence lower bound
        Used in safety test

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """  
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'ttest': 
                lower = data.mean() - stddev(data) / np.sqrt(datasize) * tinv(1.0 - delta, datasize - 1)
            else:
                raise NotImplementedError(
                    f"Bounding method {bound_method}"
                    " is not supported")
        return lower

    def compute_HC_upperbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence upper bound
        Used in safety test

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray 
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'ttest':
                upper = data.mean() + stddev(data) / np.sqrt(datasize) \
                    * tinv(1.0 - delta, datasize - 1)
            else:
                raise NotImplementedError(
                    f"Bounding method {bound_method}"
                    " is not supported")
            
        return upper
    
    def compute_HC_upper_and_lowerbound(self,
        data,
        datasize,
        delta,
        **kwargs):
        """
        Calculate high confidence lower and upper bounds
        Used in safety test.
    
        Depending on the bound_method,
        this is not always equivalent
        to calling compute_HC_lowerbound() and 
        compute_HC_upperbound() independently.

        :param data: 
            Vector containing base variable  
            evaluated at each observation in dataset
        :type data: numpy ndarray 
        :param datasize: 
            The number of observations in the safety dataset
        :type datasize: int
        :param delta: 
            Confidence level, e.g. 0.05
        :type delta: float
        """
        if 'bound_method' in kwargs:
            bound_method = kwargs['bound_method']
            if bound_method == 'ttest':
                lower = self.compute_HC_lowerbound(data=data,
                    datasize=datasize,delta=delta/2,
                    **kwargs)
                upper = self.compute_HC_upperbound(data=data,
                    datasize=datasize,delta=delta/2,
                    **kwargs)

            elif bound_method == 'manual':
                pass
            else:
                raise NotImplementedError(
                    f"Bounding method {bound_method}"
                    " is not supported")
        else:
            raise NotImplementedError("Have not implemented" 
                    "confidence bounds without the keyword bound_method")

        return lower,upper
 

class ConfusionMatrixBaseNode(BaseNode):
    def __init__(self,
        name,
        cm_true_index,
        cm_pred_index,
        lower=float('-inf'),
        upper=float('inf'),
        conditional_columns=[],
        **kwargs):
        """ A confusion matrix base node. 
        Inherits all of the attributes/methods
        of basenode and sets the i,j indices
        of the K x K confusion matrix, C:

        ::

                            Predicted labels
                        | j=0  | j=1  | ... | j=K
                        ---------------------------
                    i=0 | C_00 | C_01 | ... | C_0K|
                        |______|______|_____|_____|
            True    i=1 | C_10 | C_11 | ... | C_1K|
            labels      |______|______|_____|_____|
                    ... | ...  | ...  | ... | ... |
                        |______|______|_____|_____|
                    i=K | C_K0 | C_K1 | ... | C_KK|
                        |______|______|_____|_____|
                        
        :param name: 
            The name of the node
        :type name: str
        :param cm_true_index: 
            The index of the row in the confusion matrix. 
            Rows are the true values
        :param cm_pred_index: 
            The index of the column in the confusion matrix.
            Columns are the predicted values
        :param lower: 
            Lower confidence bound
        :type lower: float
        :param upper: 
            Upper confidence bound
        :type upper: float
        :param conditional_columns: 
            When calculating confidence bounds on a measure 
            function, condition on these columns being == 1
        :type conditional_columns: List(str)
        """
        super().__init__(
            name=name,
            lower=lower,
            upper=upper,
            conditional_columns=conditional_columns,
            **kwargs)
        self.cm_true_index = cm_true_index
        self.cm_pred_index = cm_pred_index


class MultiClassBaseNode(BaseNode):
    def __init__(self,
        name,
        class_index,
        lower=float('-inf'),
        upper=float('inf'),
        conditional_columns=[],
        **kwargs):
        """ A base node for computing 
        the classification statistic 
        for a single class against all
        other classes. For example, 
        if one has 3 classes and wants the 
        false positive rate of the first class,
        they would write "FPR_[0]" in their constraint
        and we would calculate the rate at which
        the model predicted class 0 when the true
        label was not class 0 (i.e., class 1 or 2).
        Inherits all of the attributes/methods
        of basenode 
                        
        :param name: 
            The name of the node, e.g. "FPR_[0]"
        :type name: str
        :param class_index: 
            The class index against which to calculate
            the statistic, e.g. false positive rate
        :param lower: 
            Lower confidence bound
        :type lower: float
        :param upper: 
            Upper confidence bound
        :type upper: float
        :param conditional_columns: 
            When calculating confidence bounds on a measure 
            function, condition on these columns being == 1
        :type conditional_columns: List(str)
        """
        super().__init__(
            name=name,
            lower=lower,
            upper=upper,
            conditional_columns=conditional_columns,
            **kwargs)
        self.class_index = class_index  
    
class ConstantNode(Node):
    def __init__(self,name,value,**kwargs):
        """ 
        Class for constant leaf nodes 
        in the parse tree. Sets lower and upper
        bound as the value of the constant.

        :param name: 
            The name of the node
        :type name: str
        :param value: 
            The value of the constant the node represents
        :type value: float
        :ivar node_type: 
            'constant_node'
        :vartype node_type: str
        """
        super().__init__(name=name,
            lower=value,upper=value,**kwargs)
        self.value = value
        self.node_type = 'constant_node'
  

class InternalNode(Node):
    def __init__(self,name,
        lower=float('-inf'),upper=float('inf'),**kwargs):
        """ 
        Class for internal (non-leaf) nodes 
        in the parse tree.
        These represent operators, such as +,-,*,/ etc.

        :param name: 
            The name of the node, which is the 
            string representation of the operation 
            the node performs
        :type name: str
        :param lower: 
            Lower confidence bound
        :type lower: float
        :param upper: 
            Upper confidence bound
        :type upper: float
        """
        super().__init__(name,lower,upper,**kwargs)
        self.node_type = 'internal_node'
