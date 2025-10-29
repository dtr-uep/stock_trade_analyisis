import pandas as pd
import numpy as np
import sys
from src.metrics.risk import qlike
from src.preprocess.data_transform import make_shocks, make_lags

# ARCH mannually derived
class ARCH_MD():
    '''
        q = day lags
        log: 
        - None - default:Linear with squared shocks
        - 'log': log both sides
        - 'hybrid': log variance model by weighted sum of squared shocks
    '''

    def __init__(self, q:int, epochs:int, lr=0.00001, log=None):
        self.q = q
        self.epochs = epochs
        self.lr = lr
        self.log = log

    def __loss__(self, RV, varhat):
        return ((RV + 1e-12) / (varhat + 1e-12) - np.log((RV + 1e-12) / (varhat + 1e-12)) - 1).mean()

    def __forward__(self, X):
        if self.log == None:
            varhat = X @ self.alphas
        elif self.log == 'log' or self.log == 'hybrid':
            varhat = np.exp(X @ self.alphas)
        return varhat
        
    def __backward__(self, X, varhat, RV):
        if not self.log:
            # Gradient descent
            numerator = -(RV + 1e-12) / (varhat + 1e-12)**2 + 1 / (varhat + 1e-12)
            multi = X.copy()
            multi.iloc[varhat == 0, :] = 0 # ReLU
            gradident = (multi.T @ numerator) / varhat.shape[0]
            self.alphas -= self.lr * gradident # Full batch GD considering timeseries autocorrelation

            # Make sure alphas stay positive
            self.alphas['cnst'] = max(self.alphas['cnst'], 1e-5)
            self.alphas = self.alphas.clip(lower=0)       
        
        else:
            multi = X * np.asarray(varhat).reshape(-1,1)
            gradident = (multi.T @ numerator) / varhat.shape[0]
            self.alphas -= self.lr * gradident # Alphas dont have to be positive

    def fit(self, returns:pd.Series, validation:pd.Series=None, mean:pd.Series=None, verbose:bool=0, stopstep:int=None):
        if not mean:
           self.mean = returns.mean() # Data mean if not using AV
        
        current_min_loss = None
        current_step = stopstep

        self.train_loss = []
        self.val_loss = []

        # Transform input 
        shock_df = make_shocks(returns, self.mean)**2 
        X = make_lags(shock_df, self.q) if not self.log == 'log' else np.log(make_lags(shock_df, self.q))
        X['cnst'] = 1

        shock_df_val = make_shocks(validation, self.mean)**2 
        X_val = make_lags(shock_df_valval, self.q) if not self.log == 'log' else np.log(make_lags(shock_df_val + 1e-12, self.q))
        X_val['cnst'] = 1
        
        # Realize variance
        RV = shock_df.iloc[self.q:, 0]
        RV_val = shock_df_val.iloc[self.q, 0]

        # Initialize weights
        intercept = returns.std()**2 if self.log == None else 0
        self.alphas_list = ([1/(self.q + 1)] * self.q + [intercept]) # Historical variance plus weighted past shocks
        self.alphas = np.asarray(self.alphas_list)

        # Train model
        for epoch in range(self.epochs):
            varhat = np.clip(self.__forward__(X), a_max=None, a_min=0) # ReLU for linear model
            self.__backward__(X, varhat, RV)

            loss = self.__loss__(RV, varhat)
            self.train_loss.append(loss)

            if validation is not None:
                varhat_val = np.clip(self.__forward__(X_val), a_max=None, a_min=0)
                val = self.__loss__(RV_val, varhat_val)
                self.val_loss.append(val)
            
            if current_min_loss:
                if current_min_loss >= loss:
                    current_min_loss = loss
                    current_step = stopstep
                    
                else:
                    current_step -= 1
                    if stopstep and current_step == 0:
                        break

            else:
                current_min_loss = loss

            if verbose:
                message = f"Epoch: {epoch} - Loss: {loss}"
                if validation is not None:
                    
                    message += f" - Validation: {val}"
                
                print(message)
    
    def summary(self):
        return pd.DataFrame(self.alphas).rename(columns={0:'coef value'}).sort_values(by='coef value', ascending=False)

    def predict(self, returns:pd.Series, mean:pd.Series=None, include_past_estimate=False):
        X = returns.copy()
        X.loc[returns.shape[0]] = 0
        if not mean:
           mean = self.mean

        # Transform input 
        shock_df = make_shocks(returns, mean)**2 
        X = make_lags(shock_df, self.q) if not self.log == 'log' else np.log(make_lags(shock_df, self.q))
        X['cnst'] = 1
        if not include_past_estimate:
            X = X.iloc[-1]

        return np.clip(self.__forward__(X), a_max=None, a_min=0)



# GARCH mannually derived
class GARCH_MD():
    
    def __init__(self, q:int, p:int, epochs:int, lr=0.001):
        self.q = q
        self.p = p
        self.lr = lr
        self.epochs = epochs

    def __loss__(self, RV, varhat):
        return ((RV + 1e-12) / (varhat + 1e-12) - np.log((RV + 1e-12) / (varhat + 1e-12)) - 1).mean()
    
    def __forward__(self, X, varlist):

        def calculate_var(n):
            if n == 0:
                var_past = np.asarray(varlist[-self.p:])
                varhat = X.iloc[n, :] @ self.alphas + var_past @ self.betas
                varlist.append(varhat)
                return varhat

            var_past = np.asarray(varlist[-self.p:-1] + [calculate_var(n-1)])
            varhat = X.iloc[n, :] @ self.alphas + var_past @ self.betas
            varlist.append(varhat)
            return varhat
        
        calculate_var(X.shape[0]-1)

    def __backward__(self, X, varhat, varlist, RV):
        # Calculate gradients
        numerator = -(RV + 1e-12) / (varhat + 1e-12)**2 + 1 / (varhat + 1e-12)
        numerator = numerator.reset_index(drop=True)
        multi_alphas = X.copy().reset_index(drop=True)
        multi_alphas.iloc[varhat < 0, :] = 0 # ReLU
        gradient_alphas = (multi_alphas.T @ numerator) / varhat.shape[0]

        varhat_matrix = make_lags(np.asarray(varlist), self.p)
        multi_betas = varhat_matrix.copy().reset_index(drop=True)
        multi_betas = multi_betas.reindex(numerator.index)
        multi_betas.iloc[varhat < 0, :] = 0 # ReLU
        gradient_betas = (multi_betas.T @ numerator) / varhat.shape[0]

        # Gradient descent
        self.alphas -= self.lr * gradient_alphas
        self.alphas['cnst'] = max(self.alphas['cnst'], 1e-5)
        self.alphas = self.alphas.clip(lower=0)

        self.betas -= self.lr * gradient_betas * 0.001
        self.betas = np.clip(self.betas, a_min=0, a_max=None)
        
    def fit(self, returns:pd.Series, validation:pd.Series=None, mean:pd.Series=None, verbose:bool=0, stopstep:int=None, var_list:list=None, val_var_list:list=None):
        current_min_loss = None
        current_step = stopstep
        if not var_list:
            self.var = returns.std()**2
            var_list_original = [self.var] * self.p
        else:
            var_list_original = var_list
        self.train_loss = []
        self.val_loss = []
        
        if not val_var_list:
            val_var_list_original = var_list_original
        else:
            val_var_list_original = val_var_list

        if not mean:
            mean = returns.mean()
        
        var_list = var_list_original.copy()
        val_var_list = val_var_list_original.copy()

        # Transform input
        shock_df = make_shocks(returns, mean)**2
        X_first_part = make_lags(shock_df, self.q)
        X_first_part['cnst'] = 1

        shock_df_val = make_shocks(returns, mean)**2
        X_val = make_lags(shock_df_val, self.q)
        X_val['cnst'] = 1

        # Initialize alphas and betas
        alpha_list = [1/(self.p+self.q+1)] * self.q + [var_list[-1]]
        self.alphas = np.asarray(alpha_list)
        beta_list = [1/(self.p+self.q+1)] * self.p
        self.betas = np.asarray(beta_list)

        # Realize variance
        RV = shock_df.iloc[self.q:, 0]
        RV_val = shock_df_val.iloc[self.q, 0]

        # Config sys
        sys.setrecursionlimit(returns.shape[0] + 100)

        # Train model
        for epoch in range(self.epochs):
            self.__forward__(X_first_part, var_list)
            varhat = np.asarray(var_list[self.p:])
            varhat_final = np.clip(varhat, a_max=None, a_min=0)
            self.__backward__(X_first_part, varhat, var_list, RV)
            loss = self.__loss__(RV, varhat_final)
            self.train_loss.append(loss)

            if validation is not None:
                self.__forward__(X_val, val_var_list)
                val = self.__loss__(RV_val, np.clip(np.asarray(val_var_list[self.p]), a_max=None, a_min=0))
                self.val_loss.append(val)

            var_list = var_list_original.copy()
            val_var_list = val_var_list_original.copy()

            if current_min_loss:
                if current_min_loss >= loss:
                    current_min_loss = loss
                    current_step = stopstep
                    
                else:
                    current_step -= 1
                    if stopstep and current_step == 0:
                        break

            else:
                current_min_loss = loss

            if verbose:
                message = f"Epoch: {epoch} - Loss: {loss}"
                if validation is not None:
                    
                    message += f" - Validation: {val}"
                
                print(message)


        # Reset config
        sys.setrecursionlimit(3000)

    def summary(self):
        alphas_df = pd.DataFrame(self.alphas)
        betas_index = [f"Var_lag_{n}" for n in range(self.p, 0, -1)]
        betas_df = pd.DataFrame(self.betas.values, index=betas_index)

        return pd.concat([alphas_df, betas_df], axis=0).rename(columns={0:'coef value'}).sort_values(by='coef value', ascending=False)

    def predict(self, returns:pd.Series, mean:pd.Series=None, var_list:list=None, include_past_estimate=False):
        X = returns.copy()
        X.loc[X.shape[0]] = 0
        if not var_list:
            var_list = [self.var] * self.p
        if not mean:
            mean = returns.mean()
        sys.setrecursionlimit(X.shape[0] + 100)

        # Transform input
        shock_df = make_shocks(X, mean)**2
        X_first_part = make_lags(shock_df, self.q)
        X_first_part['cnst'] = 1
        if not include_past_estimate:
            X_first_part = X_first_part.iloc[-1]

        # Calculate predction
        self.__forward__(X_first_part, var_list)
        sys.setrecursionlimit(3000)

        return np.clip(np.asarray(var_list[self.p:]), a_max=None, a_min=0)
