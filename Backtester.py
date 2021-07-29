#feature engineering: https://alphascientist.com/feature_engineering.html
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc
from multiprocessing import Pool
from time import time
from lightgbm import LGBMRegressor, LGBMClassifier
pd.set_option('display.max_columns', None)


class Strategy:
    '''
    Parent class for all specific strategies to inherit.
    '''
    def __init__(self, stonk_data, max_stonk_to_hold, preds = 'preds', yesterday_close = 'T_close_1day',
                sell_percent=4., hodl_days = 3, name = ''):
        '''
        PARAMETERS:
        ------------

        stonk_data: (df) list of all stock info for the time period to run the strategy
        max_stonk_to_hold: (int) the maximum # of stonk portfolio can have.
        preds: (str) column in df that gives the model's predictions.
        yesterday_close: (str) column in df that gives the price of yesterday's close
        sell_percent:(float) percent increase in stock price needed to sell
        hodl_days: (int) max number of days to hold stock.  Stock sold on close of max day.
                    If None, hold stocks until sell_percent reached
        name: (str) used for grouping together repeated runs of the same strategy


        CLASS VARIABLES
        ---------------

        preds: as above
        account: (int) the amount of money your model has to buy stonk.
        earnings: the total growth in the account.  Updates after every sell.  Does not change with
            1)buying stonk or 2) the bought stonk's growth or decline.
        held_stonk: the stonk the model holds currently. [{name: (str), buy_price: $, money_spent:$, buy_date: (datetime)}]
        max_stonk_to_hold: as above
        yesterday_close:as above
        positions: Dataframe giving a history of all stonk transactions by the strategy.
        today: (DateTime) today's date
        today_fake: (int) a fake day starting at -1 that counts the number of elapsed trading days.
        bankrupt_day: ({stonk: DateTime}) dictionary of the last day of all stocks in stonk_data
        '''

        self.preds = preds
        self.account = 1
        self.earnings = 1
        self.held_stonk = [] #[{name: (str), buy_price: $, money_spent:$, buy_date: (datetime), fake_date: (int), current_price:(float)}]
        self.max_stonk_to_hold = max_stonk_to_hold
        self.yesterday_close = yesterday_close
        self.positions = [] #pd.DataFrame(columns=['name', 'money_spent','buy_date', 'buy_price',
                            #                   'sell_date', 'sell_price', 'days_held', 'percent_profit'])
        self.today = stonk_data.day.min()
        self.today_fake = -1
        self.growth_chart = {'day':[], 'today_fake':[], 'account':[], 'earnings':[], 'sales':[], 'buys':[],
                             'held':[], 'portfolio_and_account':[]}
        self.sell_percent = 1 + .01 * sell_percent
        self.hodl_days = hodl_days
        self.name = name

    def daily_trading(self, daily_df, today, bankrupt_day):
        '''
        Places buy and sell orders for 1 day.

        PARAMETERS
        ----------
        daily_df: (df) list of all stock data for the day
        today: (DateTime) today's date
        '''

        self.today = today
        self.today_fake = 1 + self.today_fake
        for fun in [self.buy, self.sell]:#.shuffle(): #Uncomment the shuffle in the actual program
            buy_sell, number = fun(daily_df, bankrupt_day)
            if buy_sell == 'sell':
                num_sales = number
            else:
                num_buys = number

        #Finding the value of all stonk in my account
        portfolio = 0
        for held_stonk in self.held_stonk:
            portfolio += (held_stonk['current_price'] / held_stonk['buy_price'] )* held_stonk['money_spent']

        self.growth_chart['account'].append(self.account)
        self.growth_chart['day'].append(self.today)
        self.growth_chart['today_fake'].append(self.today_fake)
        self.growth_chart['earnings'].append(self.earnings)
        self.growth_chart['sales'].append(num_sales)
        self.growth_chart['buys'].append(num_buys)
        self.growth_chart['held'].append(len(self.held_stonk))
        self.growth_chart['portfolio_and_account'].append(self.account + portfolio)

    def sell(self, daily_df, bankrupt_day):
        '''
        Logic for placing sell orders
        '''

        sell_dict = {} #stonk: {sell_price: $, idx: int}

        if len(self.held_stonk) == 0:
            #tt(f'No held stonk')
            return 'sell', 0

        #Sell if  1) value increased by self_percent, 2) hodl_days days after purchasing, 3) bankrupt
        for idx, stonk in enumerate(self.held_stonk):


            #Is stonk bankrupt?
            if self.today >= bankrupt_day[stonk['name']]:
                sell_dict[stonk['name']] = {'sell_price':0., 'idx': idx}
                continue

            #Is stock info recorded for the day?
            daily_row = daily_df[daily_df.ticker == stonk['name']]
            if daily_row.shape[0] != 1: #If stock
                continue


            #Writing current price in self.held_stonk to keep track of portfolio size
            self.held_stonk[idx]['current_price'] = daily_df.loc[daily_df.ticker == stonk['name'], 'open'].values[0]

            #Value increased by sell_percent%
            if daily_row['high'].values[0] >= self.sell_percent * stonk['buy_price']:
                #Can only sell if you 1) didn't buy today or 2) bought today at open
                if (stonk['buy_date'] != self.today) or (stonk['buy_price'] == daily_row['open'].values[0]):
                    price = self.sell_percent * stonk['buy_price']
                    sell_dict[stonk['name']] = {'sell_price':price, 'idx': idx}

            #been hodl_days trading days
            elif self.hodl_days is not None:
                if (self.today_fake - stonk['fake_date']) >= self.hodl_days:
                    price = daily_row['close'].values[0] #see at close
                    sell_dict[stonk['name']] = {'sell_price':price, 'idx': idx}


        #print(sell_dict)
        num_sales = len(sell_dict)
        if len(sell_dict.keys()) > 0:
            self.sell_internals(sell_dict)

        return 'sell', num_sales



    def sell_internals(self, sell_dict):
        '''
        Takes the dictionary of sell orders and updates t
        self.positions: adds a row for each sale
        self.account: add the money back to the account
        self.earnings: add the sell profit or loss to earnings.

        buy_dict: {stonk: {'sell_price':$, 'idx': int} }, idx is the index
        of the held_stonk that the sell is from
        '''
        #self.positions = pd.DataFrame(columns=['stonk', 'money_spent','buy_date',
        #                                       'buy_price', 'sell_date', 'sell_price', 'days_held',
        #                                       'percent_profit'])
        old_indices = []
        for stonk in sell_dict.keys():
            #name: (str), buy_price: $, money_spent:$, buy_date: (datetime), fake_date: (int)}
            stonk_info = self.held_stonk[sell_dict[stonk]['idx']]
            old_indices.append(sell_dict[stonk]['idx'])
            stonk_info['sell_date'] = self.today
            stonk_info['sell_price'] = sell_dict[stonk]['sell_price']
            stonk_info['days_held'] = self.today_fake - stonk_info['fake_date']
            stonk_info['percent_profit'] = stonk_info['sell_price'] / stonk_info['buy_price']
            self.positions.append(pd.DataFrame({key:[stonk_info[key]] for key in stonk_info.keys()}))

            self.earnings += (stonk_info['percent_profit'] - 1) * stonk_info['money_spent']
            self.account +=  stonk_info['percent_profit'] * stonk_info['money_spent']

        #deleting old entries in self.held_stonk
        for ele in sorted(old_indices, reverse = True):
            del self.held_stonk[ele]

    def buy(self, daily_df, bankrupt_day=None):
        '''
        Logic for placing sell orders
        '''
        buy_dict = {} #{stonk: bought_price}


        #Buy the stonk if >.8 preds, low <= yesterday close close of yesterday,
        # and the stonk is among top1= 10 best predicted of day
        eleventh_best = daily_df['preds'].sort_values(ascending=False).reset_index(drop=True)[10]
        msk = (daily_df['preds'] >= .8) & (daily_df['low'] <= daily_df[self.yesterday_close]) &\
            (daily_df['open'] != 0) & (daily_df['preds'] > eleventh_best)
        for ticker, open_, yesterday_close in daily_df.loc[msk, ['ticker', 'open', self.yesterday_close]].values.tolist():
            if open_ < yesterday_close:
                buy_price = open_
            else:
                buy_price = yesterday_close
            buy_dict[ticker] = buy_price

        #print(f'buy_dict: {buy_dict}')
        num_buys = self.buy_internals(buy_dict)
        return 'buy', num_buys

    def buy_internals(self, buy_dict):
        '''
        Takes the dictionary of buy orders and updates the Strategy internals
        buy_dict: {stonk: buy_price}
        '''
        if buy_dict == {}:
            #print('No stonk found to buy')
            return 0

        max_buys = self.max_stonk_to_hold - len(self.held_stonk)#maximum number of stocks to buy
        num_of_choices = len(buy_dict.keys()) #potential stocks to be bought

        if max_buys == 0: #If you don't have any slots to buy stock, quit buying
            #print(f'Num held: {self.held_stonk}, Max in portfolio: {self.max_stonk_to_hold}')
            return 0
        price_of_stock = self.account / max_buys

        if self.account <= 0.001: #If you are out of money, quit buying
            #print(f'Account: {self.account}, cannot purchase stonk anymore')
            return 0

        #Selecting the stocks to be bought at random
        if max_buys >= num_of_choices:
            #print(f'max_buys {max_buys}>= num_of_choices {num_choices}')
            keys = list(buy_dict.keys())
        else:
            #print(f'max_buys {max_buys} < num_of_choices {num_of_choices}')
            keys = np.random.choice(list(buy_dict.keys()), size = max_buys, replace=False)

        for key in keys:
            self.held_stonk.append({'name': key, 'buy_price': buy_dict[key],
                                    'money_spent': price_of_stock,
                                    'buy_date': self.today, 'fake_date': self.today_fake})
            self.account = self.account - price_of_stock
        return len(keys)


class Daily_Mover():
    '''
    This class houses multiple Strategy objects.  It
    1) starts with a day's data
    2) runs each Strategy object on the day's data
    3) repeats 1) and 2) until all daily data done.

    Additionally, the class saves memory by creating a single bankrupt_day dataframe
    to be shared among all strategy objects.

    '''
    def __init__(self, strategy_objects, stonk_data):
        '''
        INPUTS
        --------
        strategy_objects: (list of objects class Strategy) list of strategies to be used
        stonk_data: (pd.DataFrame) dataframe of all stonk data
        '''

        #Creating dictionary of when each stonk goes bankrupt
        self.bankrupt_day = {}
        end_dates = stonk_data[stonk_data.day > '2018-01-01'].groupby('ticker').tail(1)[['ticker','day']].reset_index(drop=True)
        for ticker, day in end_dates.values.tolist():
            self.bankrupt_day[ticker] = day
        del end_dates; gc.collect()

        self.strategies = strategy_objects
        self.strategy_names = []
        for strat in self.strategies:
            self.strategy_names.append(strat.name)
        self.strategy_names = list(set(self.strategy_names))

    def get_portfolio_growth_std(self, end, start='2018-01-01'):
        '''
        Gives mean and std for portfolio growth accross all n runs
        of model.
        '''
        portfolio_dict = {}
        for i in range(len(self.strategies)):
            name = self.strategies[i].name

            if name not in portfolio_dict.keys():
                portfolio_dict[name] = []

            df = pd.DataFrame(self.strategies[i].growth_chart)
            first = df.loc[df['day']==start, 'portfolio_and_account'].values[0]
            last = df.loc[df['day']==end, 'portfolio_and_account'].values[0]
            portfolio_dict[name].append(last / first)

        stat_dict = {}
        for key in portfolio_dict.keys():
            stat_dict[key] = {'mean':np.mean(portfolio_dict[key]),
                                  'std':np.std(portfolio_dict[key])}
        return stat_dict

    def get_average_portfolio(self):
        '''
        Creates a pd.DataFrame of the average portfolio_and_account across
        all different repeats of the same strategy.
        '''
        portfolio_dict = {}
        for i in range(len(self.strategies)):
            name = self.strategies[i].name

            if name not in portfolio_dict.keys():
                portfolio_dict[name] = []

            portfolio_dict[name].append(pd.DataFrame(self.strategies[i].growth_chart)['portfolio_and_account'])

        portfolio_df = pd.DataFrame(self.strategies[i].growth_chart)[['day']]
        for key in portfolio_dict.keys():
            portfolio_df[key] = np.mean(pd.concat(portfolio_dict[key], 1).values, axis=1)

        return portfolio_df


    def test_strategies(self, daily_data, END_DATE = '2018-06-01'):
        '''
        Runs the strategies for each day sequentially.
        INPUTS
        --------
        daily_data: (pd.DataFrame) dataframe of stonks starting at START_DAY and ending on END_DAY.
        END_DATE: ('YYYY-MM-DD') final date for evaluating strategy performance
        '''

        for i, one_days_data in daily_data.groupby('day'):
            day = one_days_data.day.unique()[0]
            if day < np.datetime64(END_DATE):
                print(i)
                for strat in self.strategies:
                    strat.daily_trading(one_days_data, day, self.bankrupt_day)
            else:
                break
