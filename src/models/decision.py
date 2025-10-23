import pandas as pd
import numpy as np
from src.preprocess.data_transform import make_shocks, make_lags

# Decision model
class price_volatility():

    def __init__(self, adverse:float, cash:float, p:int, number_of_shares_original:int, discount_rate:float, explore_rate:float, decay_rate:float, lr:float, epochs:int, action_list:list, batch_size:int=None, p_var:int=0):
        self.A = adverse
        self.c = cash
        self.n = number_of_shares_original
        self.gamma = discount_rate
        self.e = explore_rate
        self.alpha = decay_rate
        self.lr = lr
        self.epochs = epochs
        self.action_list = action_list
        self.p = p
        self.p_var = p_var
        self.bs = batch_size

    def __total_capital__(self, cash, number_of_shares, t, price):
        return cash + number_of_shares * price.iloc[t+1] # +1 since capital calculate with day's end price
    
    def __check_action__(self, number_of_shares, cash, t, action, price):
        action = action[t]
        while True:
            new_number_of_shares = number_of_shares + self.action_list[action]
            new_cash = cash - self.action_list[action] * price.iloc[t]
            if new_number_of_shares < 0 or new_cash < 0:
                action -= 1
                continue

            return new_number_of_shares, new_cash, action
    
    def __reward__(self, tc1, tc2):
        r = tc2 - tc1
        return r if r > 0 else self.A* r

    def __value__(self, rewards):
        return (1- self.gamma - self.gamma**2)*rewards[0] + self.gamma*rewards[1] + self.gamma**2*rewards[2] / self.c # Scale down from the currency

    def fit(self, df:pd.DataFrame, var, verbose:bool=1):
        # Transform X
        price_whole = df['close'].iloc[self.p-1:].reset_index(drop=True)
        X_whole = make_lags(df['close'], self.p).iloc[:-self.p_var, :] if self.p_var != 0 else make_lags(df['close'], self.p) # Rerange data if use GARCH
        X_whole = X_whole.reset_index(drop=True)
        X_whole['var_pred'] = var[self.p:].reset_index(drop=True)
        self.data_mean = X_whole.mean(axis=0)
        self.data_std = X_whole.std(axis=0)
        X_whole = (X_whole - self.data_mean) / self.data_std

        if self.bs == None:
            self.bs = X_whole.shape[0]
        
        number_of_batch = X_whole.shape[0] // self.bs

        self.final_capital_list = []
        self.cash_list = []
        self.shares_list = []
        # Initialize policy weights
        self.W = np.zeros(shape=(X_whole.shape[1], len(self.action_list)))

        # Train model
        for epoch in range(self.epochs):
            number_of_shares = self.n
            cash = self.c

            # Original total capital
            TC0 = self.__total_capital__(cash, number_of_shares, -1, price_whole)

            # Train per batch
            for b in range(number_of_batch):
                price = price_whole.iloc[b*self.bs:(b+1)*self.bs+1].reset_index(drop=True)
                X = X_whole.iloc[b*self.bs:(b+1)*self.bs, :].reset_index(drop=True)

                # Initialize value buffer
                V = np.zeros(shape=(X.shape[0], len(self.action_list)))

                # Calculate policy for whole period
                q = X @ self.W
                policy_action = np.argmax(q, axis=1)

                # Take action
                explore = np.random.uniform(0, 1, policy_action.shape) < self.e
                random_action = np.random.randint(0, len(self.action_list), policy_action.shape)
                taken_action = np.where(explore, random_action, policy_action)


                # Go through actions
                for t in range(X.shape[0]-2):
                    # Real action taken
                    number_of_shares, cash, taken_action[t] = self.__check_action__(number_of_shares, cash, t, taken_action, price)
                    TC1 = self.__total_capital__(cash, number_of_shares, t, price)

                    # Next action as policy
                    number_of_shares_1, cash_1, policy_action[t+1] = self.__check_action__(number_of_shares, cash, t+1, policy_action, price)
                    TC2 = self.__total_capital__(cash_1, number_of_shares_1, t+1, price)

                    # Second next action as policy
                    number_of_shares_2, cash_2, policy_action[t+2] = self.__check_action__(number_of_shares, cash, t+2, policy_action, price)
                    TC3 = self.__total_capital__(cash_2, number_of_shares_2, t+2, price)

                    # Calculate rewards
                    rewards = [self.__reward__(TC0, TC1), self.__reward__(TC1, TC2), self.__reward__(TC2, TC3)]
                    TC0 = TC1

                    # Calculate value
                    V[t, taken_action[t]] = self.__value__(rewards) 

                # Graidient ascend
                self.W += self.lr * X.T @ V  

            # Decay explore rate
            self.e *= self.alpha

            # Check final capital
            number_of_shares, cash, _ = self.__check_action__(number_of_shares, cash, X.shape[0]-2, taken_action, price)
            number_of_shares, cash, _ = self.__check_action__(number_of_shares, cash, X.shape[0]-1, taken_action, price)
            final_TC = self.__total_capital__(cash, number_of_shares, X.shape[0]-1, price) # Sum reward eqal final_TC - TC at first so compare final_TC

            self.final_capital_list.append(final_TC)
            self.cash_list.append(cash)
            self.shares_list.append(number_of_shares)

            # Verbose
            if verbose:
                print(f"Epoch: {epoch} - Final total capital: {final_TC:.2f} - Final cash: {cash} - Final shares: {number_of_shares}")
    
    def predict_score(self, df:pd.DataFrame, var_pred):
        """
        df axis 0 size should be at least size of p_price
        var_pred is only a scalar
        """
        price = df['close'].copy()
        price.loc[price.shape[0]] = 0
        X_whole = make_lags(price, self.p).iloc[-1:].reset_index(drop=True) # Rerange data if use GARCH
        X_whole = X_whole.reset_index(drop=True)
        X_whole['var_pred'] = var_pred
        X_whole = (X_whole - self.data_mean) / self.data_std

        return X_whole @ self.W
    
    def predict_action(self, df:pd.DataFrame, var_pred):
        return self.action_list[np.argmax(self.predict_score(df, var_pred))]

