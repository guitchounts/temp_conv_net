import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import keras.backend as K

from data_helpers import pass_filter, split_data, make_timeseries_instances, timeseries_shuffler,sample_dx_uniformly
from metrics_helper import do_the_thing

from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import KFold

print('test')

class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """
    def __init__(self, loss_function=None, ## was loss_function=modified_mse
                 X=None, Y=None, sample_weights=None, beta_init=None, 
                 regularization=0.00012):
        self.regularization = regularization
        self.beta = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        
        self.X = X
        self.Y = Y
            
    
    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return(prediction)

    def model_error(self):
        error = self.loss_function(
            self.predict(self.X), self.Y, sample_weights=self.sample_weights
        )
        return(error)
    
    def l2_regularized_loss(self, beta):
        self.beta = beta
        return(self.model_error() + \
               sum(self.regularization*np.array(self.beta)**2))
    
    def fit(self, maxiter=250):        
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init)==type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.array([1]*self.X.shape[1])
        else: 
            # Use provided initial values
            pass
            
        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
            
        res = minimize(self.l2_regularized_loss, self.beta_init,
                       method='BFGS', options={'maxiter': 500})
        self.beta = res.x
        self.beta_init = self.beta







# Used to cross-validate models and identify optimal lambda
class CustomCrossValidator:
    
    """
    Cross validates arbitrary model using MAPE criterion on
    list of lambdas.
    """
    def __init__(self, X, Y, ModelClass,
                 sample_weights=None,
                 loss_function=None):
        
        self.X = X
        self.Y = Y
        self.ModelClass = ModelClass
        self.loss_function = loss_function
        self.sample_weights = sample_weights
    
    def cross_validate(self, lambdas, num_folds=10):
        """
        lambdas: set of regularization parameters to try
        num_folds: number of folds to cross-validate against
        """
        
        self.lambdas = lambdas
        self.cv_scores = []
        X = self.X
        Y = self.Y 
        
        # Beta values are not likely to differ dramatically
        # between differnt folds. Keeping track of the estimated
        # beta coefficients and passing them as starting values
        # to the .fit() operator on our model class can significantly
        # lower the time it takes for the minimize() function to run
        beta_init = None
        
        for lam in self.lambdas:
            print("Lambda: {}".format(lam))
            
            # Split data into training/holdout sets
            kf = KFold(n_splits=num_folds, shuffle=True)
            kf.get_n_splits(X)
            
            # Keep track of the error for each holdout fold
            k_fold_scores = []
            
            # Iterate over folds, using k-1 folds for training
            # and the k-th fold for validation
            f = 1
            for train_index, test_index in kf.split(X):
                # Training data
                CV_X = X[train_index,:]
                CV_Y = Y[train_index]
                CV_weights = None
                if type(self.sample_weights) != type(None):
                    CV_weights = self.sample_weights[train_index]
                
                # Holdout data
                holdout_X = X[test_index,:]
                holdout_Y = Y[test_index]
                holdout_weights = None
                if type(self.sample_weights) != type(None):
                    holdout_weights = self.sample_weights[test_index]
                
                # Fit model to training sample
                lambda_fold_model = self.ModelClass(
                    regularization=lam,
                    X=CV_X,
                    Y=CV_Y,
                    sample_weights=CV_weights,
                    beta_init=beta_init,
                    loss_function=self.loss_function
                )
                lambda_fold_model.fit()
                
                # Extract beta values to pass as beta_init 
                # to speed up estimation of the next fold
                beta_init = lambda_fold_model.beta
                
                # Calculate holdout error
                fold_preds = lambda_fold_model.predict(holdout_X)
                fold_mape = modified_mse(
                    holdout_Y, fold_preds, sample_weights=holdout_weights
                )
                k_fold_scores.append(fold_mape)
                print("Fold: {}. Error: {}".format( f, fold_mape))
                f += 1
            
            # Error associated with each lambda is the average
            # of the errors across the k folds
            lambda_scores = np.mean(k_fold_scores)
            print("LAMBDA AVERAGE: {}".format(lambda_scores))
            self.cv_scores.append(lambda_scores)
        
        # Optimal lambda is that which minimizes the cross-validation error
        self.lambda_star_index = np.argmin(self.cv_scores)
        self.lambda_star = self.lambdas[self.lambda_star_index]
        print("\n\n**OPTIMAL LAMBDA: {}**".format(self.lambda_star))

def modified_mse(y_true, y_pred): #### modified MSE loss function for absolute yaw data (0-360 values wrap around)
    

    #y_true = y_true * y_std + y_mean ### y_train_mean,y_train_std are GLOBALS ??? 
    #y_pred = y_pred * y_std + y_mean

    mod_square =  np.square(np.abs(y_pred - y_true) - 360) ### hack 2.1 = (360 - np.mean(ox)) / np.std(ox) 2.1086953197291871
    raw_square =  np.square(y_pred - y_true)
    better = np.min(mod_square,raw_square)
    return np.mean(better,axis= -1)


def objective_function(beta, X, Y):
    error = loss_function(np.matmul(X,beta), Y)
    return(error)






# loss_function = modified_mse

# beta_init = np.array([1]*X.shape[1])
# result = minimize(objective_function, beta_init, args=(X,Y),
#                   method='BFGS', options={'maxiter': 500})

# # The optimal values for the input parameters are stored
# # in result.x
# beta_hat = result.x
# print(beta_hat)



# # User must specify lambdas over which to search
# lambdas = [0.1, 1.0, 10.0] #  [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# cross_validator = CustomCrossValidator(
#     X, Y, CustomLinearModel,
#     loss_function=modified_mse
# )
# cross_validator.cross_validate(lambdas, num_folds=5)


# lambda_star = cross_validator.lambda_star
# final_model = CustomLinearModel(
#     loss_function=modified_mse,
#     X=X, Y=Y, regularization=lambda_star
# )
# final_model.fit()
# final_model.beta




