import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from portfolio_analyzer.stocks_data_loader import yahoo2pandas
from dataclasses import dataclass


@dataclass
class PortfolioVisualizer:
    data : pd.DataFrame

    def plot_pie_portfolio(self):
        df = self.data

        inner_circle = plt.Circle( (0,0), 0.7, color='white')
        plt.figure(figsize=(20,10))
        plt.rcParams['axes.labelsize'] = 20
        sns.set(font_scale = 2)
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['font.size'] = 20
        plt.pie(df['weights'], labels =df['ticker'], autopct='%.0f%%',
                wedgeprops = { 'linewidth' : 5, 'edgecolor' : 'white' })
        p = plt.gcf()
        p.gca().add_artist(inner_circle)

        return plt.show()


    def benchmark_portfolio(self,
            from_date: str='2000-01-01', to_date: str='2100-12-31',
            frecuency: str='daily', bench: str='^GSPC', allow_null=True):
        """Evaluate how your portfolio perform in contrast with for example s&p index.
        """
        df = self.data
        tickers = df.ticker.to_list()
        prices_tk = yahoo2pandas(tickers, from_date, to_date,
                frecuency, allow_null)
        prices_tk = prices_tk.fillna(0)
        bench_data = yahoo2pandas(bench, from_date, to_date, frecuency)

        # I want to multiply each price by the weights of each ticker.
        # Then get the comulative for all the portfolio by day.
        for t in tickers:
            prices_tk[t] = prices_tk[t] * df.loc[df.ticker==t]['weights'].to_list()[0]

        prices_tk['cum'] = prices_tk.sum(axis=1)

        # Plotting the data
        plt.figure(figsize=(18,8))
        plt.xticks(rotation=45)
        plt.xlabel('Dates')
        plt.ylabel('Cumulative Returns')
        plt.title(f'Benchmark performance from: {from_date} to {to_date}')
        sns.lineplot(data=prices_tk['cum'], color='r', label='aw')
        sns.lineplot(data=bench_data[bench], label=bench)
        return plt.show()

    @classmethod
    def load_porfolio(cls, data_path: str):
        df = pd.read_csv(data_path)
        return PortfolioVisualizer(df)

    def dump_portfolio(self, path_name: str='./portfolio.csv'):
        """Save your portfolio data as csv file to avoid write it again the nex time.
        """
        self.data.to_csv(path_name)
