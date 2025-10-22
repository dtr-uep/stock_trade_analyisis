# Imports
import numpy as np
import pandas as pd
import math
from scipy.stats.mstats import gmean

# Configs
NUM_DAYS_YEAR = 252


# Functions
def gen_prob_set(num_combinations, num_dimensions):  

    alpha = np.ones(num_dimensions)
    probability_sets = np.random.dirichlet(alpha, size=num_combinations)

    assert np.all(probability_sets >= 0)
    assert np.allclose(probability_sets.sum(axis=1), 1)

    return probability_sets.tolist()


# Objects
class Preprocess:
    def __init__(self, data, num_days_year=NUM_DAYS_YEAR):
        self.data = data
        self.num_days_year = num_days_year
        self.means = self.data['return'].mean()
        self.geomeans_year = (gmean((self.data['return']+1)) - 1) * self.num_days_year
        self.std_year = self.data['return'].std() * math.sqrt(self.num_days_year)

    def get_report(self):
        return pd.DataFrame({
            'Mean': self.means,
            'Geomean': self.geomeans_year,
            'Std': self.std_year
        }, index=[0])
    
    def get_variables(self):
        return self.std_year, self.geomeans_year
    

class Weight_2Stocks_Portfolio:
    def __init__(self, data1, data2, weights_list, rf):
        self.rf = rf
        self.data1 = data1
        self.data2 = data2
        self.weights_list = weights_list
        # Get mean and std
        self.data1_std, self.data1_geomean = Preprocess(self.data1).get_variables()
        self.data2_std, self.data2_geomean = Preprocess(self.data2).get_variables()
        self.corel = self.data1['return'].corr(self.data2['return'])
        self.ex_var = []
        self.ex_return = []

        self.__get_portfolio_var__()
        self.__get_portfolio_return__()
        
        self.ex_std = [math.sqrt(var) for var in self.ex_var]
        self.__get_min__()
        self.__get_optimize__()
    
    def __get_portfolio_var__(self):
        for [w1, w2] in self.weights_list:
            self.ex_var.append(w1**2 * self.data1_std**2 + w2**2 * self.data2_std**2 + 2 * w1 * w2 * self.corel * self.data1_std * self.data2_std)  
    
    def __get_portfolio_return__(self):
        for [w1, w2] in self.weights_list:
            self.ex_return.append(w1 * self.data1_geomean + w2 * self.data2_geomean)
    
    def __get_min__(self):
        E1 = self.data1_geomean
        E2 = self.data2_geomean
        s1 = self.data1_std
        s2 = self.data2_std
        
        numerator = s2**2 - s1 * s2 * self.corel
        denominator = s1**2 + s2**2 - 2 * s1 * s2 * self.corel

        self.w1_min = numerator / denominator
        self.w2_min = 1 - self.w1_min
        self.Ep_min = self.w1_min * E1 + self.w2_min * E2
        self.stdp_min = math.sqrt(self.w1_min**2 * s1**2 + self.w2_min**2 * s2**2 + 2 * self.w1_min * self.w2_min * self.corel * s1 * s2)

    def __get_optimize__(self):
        E1 = self.data1_geomean
        E2 = self.data2_geomean
        s1 = self.data1_std
        s2 = self.data2_std

        numerator = (E1 - self.rf) * s2**2 - (E2 - self.rf) * s1 * s2 * self.corel
        denominator = (E1 - self.rf) * s2**2 + (E2 - self.rf) * s1**2 - (E1 + E2 - 2 * self.rf) * s1 * s2 * self.corel

        self.w1_opt = numerator / denominator
        self.w2_opt = 1 - self.w1_opt
        self.Ep_opt = self.w1_opt * E1 + self.w2_opt * E2
        self.stdp_opt = math.sqrt(self.w1_opt**2 * s1**2 + self.w2_opt**2 * s2**2 + 2 * self.w1_opt * self.w2_opt * self.corel * s1 * s2)

    def min_CAL(self, x):
        self.sharpe_ratio_min = (self.Ep_min - self.rf) / self.stdp_min
        return self.rf + self.sharpe_ratio_min * x

    def opt_CAL(self, x):
        self.sharpe_ratio_opt = (self.Ep_opt - self.rf) / self.stdp_opt
        return self.rf + self.sharpe_ratio_opt * x

    def get_report(self):
        return pd.DataFrame({
            'Var': self.ex_var,
            'Std': self.ex_std,
            'Return': self.ex_return,
            'Weights': self.weights_list
        })
        

class Weight_3Stocks_Portfolio:
    def __init__(self, data1, data2, data3, weights_list):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.weights_list = weights_list
        self.data1_std, self.data1_geomean = Preprocess(self.data1).get_variables()
        self.data2_std, self.data2_geomean = Preprocess(self.data2).get_variables()
        self.data3_std, self.data3_geomean = Preprocess(self.data3).get_variables()
        self.corel12 = self.data1['return'].corr(self.data2['return'])
        self.corel23 = self.data2['return'].corr(self.data3['return'])
        self.corel31 = self.data3['return'].corr(self.data1['return'])
        self.ex_var = []
        self.ex_return = []

        self.__get_portfolio_var__()
        self.__get_portfolio_return__()
        
        self.ex_std = [math.sqrt(var) for var in self.ex_var]
    
    def __get_portfolio_var__(self):
        for [w1, w2, w3] in self.weights_list:
            self.ex_var.append(w1**2 * self.data1_std**2 \
                                + w2**2 * self.data2_std**2 \
                                + w3**2 * self.data3_std**2 \
                                + 2 * w1 * w2 * self.corel12 * self.data1_std * self.data2_std \
                                + 2 * w2 * w3 * self.corel23 * self.data2_std * self.data3_std \
                                + 2 * w3 * w1 * self.corel31 * self.data3_std * self.data1_std 
                                )  
    
    def __get_portfolio_return__(self):
        for [w1, w2, w3] in self.weights_list:
            self.ex_return.append(w1 * self.data1_geomean + w2 * self.data2_geomean + w3 * self.data3_geomean)
    
    def get_report(self):
        return pd.DataFrame({
            'Var': self.ex_var,
            'Std': self.ex_std,
            'Return': self.ex_return,
            'Weights': self.weights_list
        })
    

class Weight_3Stocks_Portfolio:
    def __init__(self, data1, data2, data3, weights_list):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.weights_list = weights_list
        self.data1_std, self.data1_geomean = Preprocess(self.data1).get_variables()
        self.data2_std, self.data2_geomean = Preprocess(self.data2).get_variables()
        self.data3_std, self.data3_geomean = Preprocess(self.data3).get_variables()
        self.corel12 = self.data1['return'].corr(self.data2['return'])
        self.corel23 = self.data2['return'].corr(self.data3['return'])
        self.corel31 = self.data3['return'].corr(self.data1['return'])
        self.ex_var = []
        self.ex_return = []

        self.__get_portfolio_var__()
        self.__get_portfolio_return__()
        
        self.ex_std = [math.sqrt(var) for var in self.ex_var]
    
    def __get_portfolio_var__(self):
        for [w1, w2, w3] in self.weights_list:
            self.ex_var.append(w1**2 * self.data1_std**2 \
                                + w2**2 * self.data2_std**2 \
                                + w3**2 * self.data3_std**2 \
                                + 2 * w1 * w2 * self.corel12 * self.data1_std * self.data2_std \
                                + 2 * w2 * w3 * self.corel23 * self.data2_std * self.data3_std \
                                + 2 * w3 * w1 * self.corel31 * self.data3_std * self.data1_std 
                                )  
    
    def __get_portfolio_return__(self):
        for [w1, w2, w3] in self.weights_list:
            self.ex_return.append(w1 * self.data1_geomean + w2 * self.data2_geomean + w3 * self.data3_geomean)
    
    def get_report(self):
        return pd.DataFrame({
            'Var': self.ex_var,
            'Std': self.ex_std,
            'Return': self.ex_return,
            'Weights': self.weights_list
        })
    

class Weight_nStocks_Portfolio:
    def __init__(self, portfolio, weights_list, rf):

        self.weights_list = weights_list
        self.num_stock = len(portfolio)
        self.portfolio = portfolio
        self.var_first_half = 0
        self.var_second_half = 0
        self.len_list = len(self.weights_list)

        self.rf = rf

        self.ex_var = []
        self.ex_return = []

        self.__get_portfolio_var__()
        self.__get_portfolio_return__()
        
        self.ex_std = [math.sqrt(var) for var in self.ex_var]

        self.__get_Sharpe_ratio__()

    
    def __get_portfolio_var__(self):
        for i in range(self.len_list):
            self.var_first_half = 0
            self.var_second_half = 0
            for j in range(self.num_stock):
                self.var_first_half += self.weights_list[i][j] **2 * Preprocess(self.portfolio[j]).get_variables()[0]**2

                for j2 in range(self.num_stock):
                    if j == j2:
                        continue
                    self.var_second_half += self.weights_list[i][j] * self.weights_list[i][j2] * Preprocess(self.portfolio[j]).get_variables()[0]
            
            self.ex_var.append(self.var_first_half + self.var_second_half)
        

    def __get_portfolio_return__(self):

        for i in range(self.len_list):
            ex_return = 0
            for j in range(self.num_stock):
                ex_return += (self.weights_list[i][j] * Preprocess(self.portfolio[j]).get_variables()[1])
            self.ex_return.append(ex_return)
    
    def __get_Sharpe_ratio__(self):
        self.sharpe_ratio = [((self.ex_return[i] - self.rf) / self.ex_std[i]) for i in range(self.len_list)]

    def get_report(self):

        return pd.DataFrame({
            'Var': self.ex_var,
            'Std': self.ex_std,
            'Return': self.ex_return,
            'Weights': self.weights_list,
            'Sharpe': self.sharpe_ratio
        })