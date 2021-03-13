import json
from td.client import TDClient

class AmeritradeRebalanceUtils:

    def __init__(self):
        self.session = None
        self.account = None
        self.account_id = None

    def auth(self, credentials_path='./td_state.json', client_path='./td_client_auth.json'):
        with open(client_path) as f:
            data = json.load(f)
        self.session = TDClient(
            client_id=data['client_id'],
            redirect_uri=data['callback_url'],
            credentials_path=credentials_path
        )
        self.session.login()
        
        # assuming only 1 account under management
        self.account = self.session.get_accounts(fields=['positions'])[0]
        self.account_id = self.account['securitiesAccount']['accountId']
        return self.session
    
    def get_portfolio(self):
        positions = self.account['securitiesAccount']['positions']
        portfolio = {}
        for position in positions:
            portfolio[position['instrument']['symbol']] = position['longQuantity']
        return portfolio

    def place_orders_dry_run(self, portfolio_diff: dict):
        result = portfolio_diff.copy()
        prices = self._get_last_prices(result)
        for ticker, qty in portfolio_diff.items():
            round_qty = round(qty)
            abs_rounded_qty = abs(round_qty)
            result[ticker] = {
            'instruction': ('BUY' if qty > 0 else 'SELL'),
            'qty': abs_rounded_qty,
            'money_movement': round_qty*prices[ticker]*-1
            }
        return result

    def place_orders(self, place_orders_dry_run: dict):
        result = []
        for ticker, order in place_orders_dry_run.items():
            res = self.session.place_order(account=self.account_id, order=self._get_market_order_payload(ticker, order['qty'], order['instruction']))
            result.append(res)
        return result

    def _get_market_order_payload(self, ticker, quantity, instruction='BUY'):
        return {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                    "symbol": ticker,
                    "assetType": "EQUITY"
                    }
                }
            ]
        }

    def _get_last_prices(self, portfolio: dict):
        quotes = self.session.get_quotes(instruments=portfolio.keys())
        portfolio_prices = portfolio.copy()
        for ticker, _ in portfolio_prices.items():
            portfolio_prices[ticker] = quotes[ticker]['lastPrice']
        return portfolio_prices