"""
Optimized Metrics Calculator Module
Calculates essential trading performance metrics with improved data fetching and reduced complexity.

Focus areas:
- Core performance metrics (PnL, win rate, profit factor, Sharpe, Sortino)
- Risk management metrics (drawdown, largest win/loss)
- Position analytics (open/closed positions with unrealized P&L)
- Essential persona traits

Removed:
- Redundant market metrics
- Complex score alignment calculations
- Unnecessary behavioral metrics
- Overly detailed bucketing
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List
import logging


class TradingMetricsCalculator:
    """Calculate essential trading metrics with reliable data fetching"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config.get('metrics', {}).get('risk_free_rate', 0.05)
        self.trading_days = config.get('metrics', {}).get('trading_days_per_year', 252)
        self.logger = logging.getLogger(__name__)

    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate all essential trading metrics"""
        if df.empty:
            return self._empty_metrics()

        df = self._prepare_dataframe(df)
        
        metrics = {
            # Core Performance
            'total_trades': len(df),
            'total_pnl': self.calculate_total_pnl(df),
            'win_rate': self.calculate_win_rate(df),
            'avg_win': self.calculate_avg_win(df),
            'avg_loss': self.calculate_avg_loss(df),
            'profit_factor': self.calculate_profit_factor(df),
            
            # Risk Metrics
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'sortino_ratio': self.calculate_sortino_ratio(df),
            'max_drawdown': self.calculate_max_drawdown(df),
            'max_drawdown_pct': self.calculate_max_drawdown_pct(df),
            
            # Trade Analysis
            'largest_win': self.calculate_largest_win(df),
            'largest_loss': self.calculate_largest_loss(df),
            'consecutive_wins': self.calculate_consecutive_wins(df),
            'consecutive_losses': self.calculate_consecutive_losses(df),
            'avg_trade_value': self.calculate_avg_trade_value(df),
            
            # Time Analysis
            'date_range': self.get_date_range(df),
            'trading_days': self.get_trading_days(df),
            'avg_trades_per_day': self.calculate_avg_trades_per_day(df),
            'avg_holding_period': self.calculate_avg_holding_period(df),
            
            # Timeline
            'pnl_timeline': self._build_pnl_timeline(df),
        }

        # Position Analytics
        positions_data = self._compute_positions_snapshot(df)
        metrics.update(positions_data['aggregates'])
        metrics['open_positions'] = positions_data['positions']
        metrics['closed_positions'] = positions_data['closed_positions']
        metrics['gainer'] = positions_data['gainer']
        metrics['loser'] = positions_data['loser']
        metrics['symbol_details'] = positions_data['symbol_details']

        # Persona Traits
        persona_traits = self.calculate_persona_traits(df)
        metrics.update(persona_traits)

        return metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'pnl_timeline': {'dates': [], 'values': []},
            'persona_type': 'N/A',
            'trait_summary': 'No data',
            'persona_traits': {}
        }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate dataframe columns"""
        df = df.copy()
        
        # Normalize transaction type
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].astype(str).str.upper()
            df['transaction_type'] = df['transaction_type'].replace({'SELL': 'SALE'})
        
        # Ensure date column is datetime
        if 'trade_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        
        # Add trade_hour if not present
        if 'trade_date' in df.columns and 'trade_hour' not in df.columns:
            df['trade_hour'] = df['trade_date'].dt.hour
        
        return df

    # =========================================================
    # Core Performance Metrics
    # =========================================================
    
    def calculate_total_pnl(self, df: pd.DataFrame) -> float:
        """Calculate total profit and loss"""
        if 'pnl' not in df.columns:
            self.logger.warning("'pnl' column not found")
            return 0.0
        return float(df['pnl'].sum())

    def calculate_win_rate(self, df: pd.DataFrame) -> float:
        """Calculate win rate percentage"""
        if len(df) == 0 or 'pnl' not in df.columns:
            return 0.0
        winning_trades = len(df[df['pnl'] > 0])
        return float(winning_trades / len(df) * 100)

    def calculate_avg_win(self, df: pd.DataFrame) -> float:
        """Calculate average winning trade"""
        if 'pnl' not in df.columns:
            return 0.0
        winning_trades = df[df['pnl'] > 0]['pnl']
        return float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0

    def calculate_avg_loss(self, df: pd.DataFrame) -> float:
        """Calculate average losing trade"""
        if 'pnl' not in df.columns:
            return 0.0
        losing_trades = df[df['pnl'] < 0]['pnl']
        return float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

    def calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if 'pnl' not in df.columns:
            return 0.0
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    # =========================================================
    # Risk Metrics
    # =========================================================
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if len(df) < 2 or 'pnl' not in df.columns or 'trade_value' not in df.columns:
            return 0.0
        
        returns = df['pnl'] / df['trade_value'].abs()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess = returns.mean() - (self.risk_free_rate / self.trading_days)
        return float(excess / returns.std() * np.sqrt(self.trading_days))

    def calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(df) < 2 or 'pnl' not in df.columns or 'trade_value' not in df.columns:
            return 0.0
        
        returns = df['pnl'] / df['trade_value'].abs()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        downside = returns[returns < 0]
        
        if len(downside) == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        if downside.std() == 0:
            return 0.0
        
        excess = returns.mean() - (self.risk_free_rate / self.trading_days)
        return float(excess / downside.std() * np.sqrt(self.trading_days))

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown in absolute terms"""
        if 'trade_date' not in df.columns or 'pnl' not in df.columns:
            return 0.0
        
        df_sorted = df.sort_values('trade_date')
        cum_pnl = df_sorted['pnl'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        return float(drawdown.min())

    def calculate_max_drawdown_pct(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage"""
        if df.empty or 'pnl' not in df.columns or 'trade_date' not in df.columns:
            return 0.0
        
        df_sorted = df.sort_values('trade_date')
        cum_pnl = df_sorted['pnl'].cumsum()
        running_max = cum_pnl.cummax().replace(0, np.nan)
        dd_pct = (cum_pnl - running_max) / running_max.abs() * 100
        return round(abs(float(dd_pct.min())), 2) if not dd_pct.isna().all() else 0.0

    # =========================================================
    # Trade Analysis
    # =========================================================
    
    def calculate_largest_win(self, df: pd.DataFrame) -> float:
        """Calculate largest winning trade"""
        if 'pnl' not in df.columns or len(df) == 0:
            return 0.0
        return float(df['pnl'].max())

    def calculate_largest_loss(self, df: pd.DataFrame) -> float:
        """Calculate largest losing trade"""
        if 'pnl' not in df.columns or len(df) == 0:
            return 0.0
        return float(df['pnl'].min())

    def calculate_consecutive_wins(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive wins"""
        if 'trade_date' not in df.columns or 'pnl' not in df.columns:
            return 0
        
        df_sorted = df.sort_values('trade_date')
        max_consecutive = current = 0
        for pnl in df_sorted['pnl']:
            if pnl > 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return int(max_consecutive)

    def calculate_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        if 'trade_date' not in df.columns or 'pnl' not in df.columns:
            return 0
        
        df_sorted = df.sort_values('trade_date')
        max_consecutive = current = 0
        for pnl in df_sorted['pnl']:
            if pnl < 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return int(max_consecutive)

    def calculate_avg_trade_value(self, df: pd.DataFrame) -> float:
        """Calculate average trade value"""
        if 'trade_value' not in df.columns:
            return 0.0
        return float(df['trade_value'].abs().mean())

    # =========================================================
    # Time Analysis
    # =========================================================
    
    def get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range of trading period"""
        if 'trade_date' not in df.columns:
            return "N/A"
        start_date = df['trade_date'].min().strftime('%Y-%m-%d')
        end_date = df['trade_date'].max().strftime('%Y-%m-%d')
        return f"{start_date} to {end_date}"

    def get_trading_days(self, df: pd.DataFrame) -> int:
        """Get number of unique trading days"""
        if 'trade_date' not in df.columns:
            return 0
        return int(df['trade_date'].dt.date.nunique())

    def calculate_avg_trades_per_day(self, df: pd.DataFrame) -> float:
        """Calculate average trades per day"""
        if 'trade_date' not in df.columns:
            return 0.0
        days = df['trade_date'].dt.date.nunique()
        return float(len(df) / days) if days > 0 else 0.0

    def calculate_avg_holding_period(self, df: pd.DataFrame) -> float:
        """Calculate average holding period in minutes"""
        if 'holding_period_minutes' in df.columns:
            return float(df['holding_period_minutes'].mean())
        return 0.0

    # =========================================================
    # Position Analytics
    # =========================================================
    
    def _compute_positions_snapshot(self, df: pd.DataFrame) -> Dict:
        """
        Compute symbol-level positions with realized and unrealized P&L
        """
        # Get latest trading date
        latest_date = df['trade_date'].dt.date.max() if 'trade_date' in df.columns else None
        
        # Calculate signed quantities (BUY = +, SALE = -)
        if 'transaction_type' not in df.columns or 'quantity' not in df.columns:
            return self._empty_positions()
        
        q_sign = df['transaction_type'].map({'BUY': 1, 'SALE': -1}).fillna(0)
        df['_signed_qty'] = df['quantity'] * q_sign
        
        # Symbol-level aggregation
        symbol_groups = df.groupby('symbol')
        
        # Net quantities and average costs
        symbol_data = pd.DataFrame({
            'net_qty': symbol_groups['_signed_qty'].sum(),
            'avg_cost': self._calculate_avg_cost(df, symbol_groups),
            'last_price': symbol_groups['price'].last()
        })
        
        # Get realized P&L by symbol
        realized_by_symbol = symbol_groups['pnl'].sum() if 'pnl' in df.columns else pd.Series()
        
        # Separate open and closed positions
        open_positions = symbol_data[symbol_data['net_qty'] != 0].copy()
        closed_positions = symbol_data[symbol_data['net_qty'] == 0].copy()
        
        # Calculate unrealized P&L for open positions
        if not open_positions.empty:
            open_positions['invested_value'] = open_positions['net_qty'].abs() * open_positions['avg_cost']
            open_positions['unrealized'] = (
                (open_positions['last_price'] - open_positions['avg_cost']) * 
                open_positions['net_qty']
            )
            open_positions['pct_change'] = (
                open_positions['unrealized'] / open_positions['invested_value'] * 100
            ).fillna(0)
        
        # Calculate aggregates
        total_realized = float(realized_by_symbol.sum()) if len(realized_by_symbol) > 0 else 0.0
        total_unrealized = float(open_positions['unrealized'].sum()) if not open_positions.empty else 0.0
        total_investment = float(open_positions['invested_value'].sum()) if not open_positions.empty else 0.0
        
        # Day MTM (if latest date available)
        day_mtm = 0.0
        if latest_date and 'trade_date' in df.columns and 'pnl' in df.columns:
            day_mtm = float(df.loc[df['trade_date'].dt.date == latest_date, 'pnl'].sum())
        
        # Gainers and Losers
        symbol_totals = []
        for symbol in realized_by_symbol.index:
            realized = float(realized_by_symbol.get(symbol, 0.0))
            unrealized = float(open_positions.loc[symbol, 'unrealized']) if symbol in open_positions.index else 0.0
            symbol_totals.append((symbol, realized + unrealized))
        
        gainers = sorted([s for s, pnl in symbol_totals if pnl > 0])
        losers = sorted([s for s, pnl in symbol_totals if pnl < 0])
        
        # Build position lists
        positions_list = []
        for symbol, row in open_positions.iterrows():
            positions_list.append({
                'symbol': symbol,
                'net_qty': int(row['net_qty']),
                'avg_cost': float(row['avg_cost']),
                'last_price': float(row['last_price']),
                'invested_value': float(row['invested_value']),
                'unrealized': float(row['unrealized']),
                'pct_change': float(row['pct_change'])
            })
        
        closed_list = []
        for symbol, row in closed_positions.iterrows():
            realized_pnl = float(realized_by_symbol.get(symbol, 0.0))
            total_value = float(df[df['symbol'] == symbol]['trade_value'].abs().sum())
            closed_list.append({
                'symbol': symbol,
                'realized_pnl': realized_pnl,
                'pct_change': (realized_pnl / total_value * 100) if total_value > 0 else 0.0
            })
        
        # Symbol details for UI
        symbol_details = {}
        for symbol in set(list(realized_by_symbol.index) + list(open_positions.index)):
            is_open = symbol in open_positions.index
            
            if is_open:
                row = open_positions.loc[symbol]
                symbol_details[symbol] = {
                    'buy_rate': float(row['avg_cost']),
                    'sell_rate': float(row['last_price']),
                    'qty': int(row['net_qty']),
                    'value': float(row['invested_value']),
                    'pnl': float(row['unrealized']),
                    'return_pct': float(row['pct_change'])
                }
            else:
                realized_pnl = float(realized_by_symbol.get(symbol, 0.0))
                total_value = float(df[df['symbol'] == symbol]['trade_value'].abs().sum())
                symbol_details[symbol] = {
                    'buy_rate': 0.0,
                    'sell_rate': 0.0,
                    'qty': 0,
                    'value': total_value,
                    'pnl': realized_pnl,
                    'return_pct': (realized_pnl / total_value * 100) if total_value > 0 else 0.0
                }
        
        return {
            'aggregates': {
                'total_trades': len(df),
                'day_mtm': day_mtm,
                'total_realized_pnl': total_realized,
                'total_unrealized_pnl': total_unrealized,
                'total_pnl_combined': total_realized + total_unrealized,
                'total_investment_value_open': total_investment,
                'open_positions_count': len(open_positions),
            },
            'positions': positions_list,
            'closed_positions': closed_list,
            'gainer': {'count': len(gainers), 'list': gainers},
            'loser': {'count': len(losers), 'list': losers},
            'symbol_details': symbol_details
        }
    
    def _calculate_avg_cost(self, df: pd.DataFrame, symbol_groups) -> pd.Series:
        """Calculate weighted average cost for each symbol"""
        result = {}
        for symbol in symbol_groups.groups.keys():
            symbol_df = df[df['symbol'] == symbol]
            buys = symbol_df[symbol_df['transaction_type'] == 'BUY']
            if len(buys) > 0:
                total_cost = (buys['price'] * buys['quantity']).sum()
                total_qty = buys['quantity'].sum()
                result[symbol] = total_cost / total_qty if total_qty > 0 else 0.0
            else:
                result[symbol] = 0.0
        return pd.Series(result)
    
    def _empty_positions(self) -> Dict:
        """Return empty positions structure"""
        return {
            'aggregates': {
                'total_trades': 0,
                'day_mtm': 0.0,
                'total_realized_pnl': 0.0,
                'total_unrealized_pnl': 0.0,
                'total_pnl_combined': 0.0,
                'total_investment_value_open': 0.0,
                'open_positions_count': 0,
            },
            'positions': [],
            'closed_positions': [],
            'gainer': {'count': 0, 'list': []},
            'loser': {'count': 0, 'list': []},
            'symbol_details': {}
        }

    def _build_pnl_timeline(self, df: pd.DataFrame) -> Dict[str, List]:
        """Build cumulative P&L timeline for charts"""
        if 'trade_date' not in df.columns or 'pnl' not in df.columns:
            return {'dates': [], 'values': []}
        
        df_sorted = df.sort_values('trade_date')
        daily_pnl = df_sorted.groupby(df_sorted['trade_date'].dt.date)['pnl'].sum()
        cumulative = daily_pnl.cumsum()
        
        return {
            'dates': [str(d) for d in cumulative.index],
            'values': [float(v) for v in cumulative.values]
        }

    # =========================================================
    # Simplified Persona Traits
    # =========================================================
    
    def calculate_persona_traits(self, df: pd.DataFrame) -> Dict:
        """Calculate simplified persona traits"""
        if df.empty:
            return {'persona_type': 'N/A', 'trait_summary': 'No data', 'persona_traits': {}}
        
        traits = {
            'discipline_score': self._calc_discipline_score(df),
            'risk_appetite': self._calc_risk_appetite(df),
            'consistency': self._calc_consistency(df),
        }
        
        persona_type = self._map_persona(traits)
        trait_summary = self._summarize_persona(traits, persona_type)
        
        return {
            'persona_type': persona_type,
            'trait_summary': trait_summary,
            'persona_traits': traits
        }
    
    def _calc_discipline_score(self, df: pd.DataFrame) -> float:
        """Calculate discipline based on trade size consistency"""
        if 'quantity' not in df.columns or 'pnl' not in df.columns:
            return 0.5
        
        qty_std = np.std(df['quantity'])
        qty_mean = np.mean(df['quantity'])
        pnl_std = np.std(df['pnl'])
        
        trade_size_var = qty_std / (qty_mean + 1e-6)
        discipline = 1 / (1 + 0.5 * trade_size_var + 0.5 * (pnl_std / 1000))
        return round(max(0.0, min(1.0, discipline)), 2)
    
    def _calc_risk_appetite(self, df: pd.DataFrame) -> float:
        """Calculate risk appetite based on trade values and volatility"""
        if 'trade_value' not in df.columns or 'pnl' not in df.columns:
            return 0.5
        
        avg_trade_value = df['trade_value'].mean()
        pnl_std = np.std(df['pnl'])
        max_dd = abs(self.calculate_max_drawdown(df))
        
        raw_score = (pnl_std + avg_trade_value * 0.001) / (max_dd + 1e-6)
        return round(min(1.0, max(0.0, raw_score * 5)), 2)
    
    def _calc_consistency(self, df: pd.DataFrame) -> float:
        """Calculate consistency based on P&L volatility"""
        if 'pnl' not in df.columns or 'trade_date' not in df.columns:
            return 0.5
        
        daily_pnl = df.groupby(df['trade_date'].dt.date)['pnl'].sum()
        volatility = daily_pnl.std()
        mean_pnl = daily_pnl.mean()
        
        consistency = 1 - abs(volatility / (abs(mean_pnl) + 1))
        return round(max(0.0, min(1.0, consistency)), 2)
    
    def _map_persona(self, traits: Dict[str, float]) -> str:
        """Map traits to persona type"""
        risk = traits['risk_appetite']
        discipline = traits['discipline_score']
        consistency = traits['consistency']
        
        if risk > 0.7 and discipline < 0.5:
            return "Aggressive Trader"
        elif discipline > 0.7 and consistency > 0.7:
            return "Disciplined Trader"
        elif risk < 0.5 and consistency > 0.6:
            return "Conservative Trader"
        else:
            return "Balanced Trader"
    
    def _summarize_persona(self, traits: Dict[str, float], persona_type: str) -> str:
        """Generate persona summary"""
        return (
            f"Trader profile: **{persona_type}**\n\n"
            f"- Discipline: {traits['discipline_score']*100:.0f}%\n"
            f"- Risk Appetite: {traits['risk_appetite']*100:.0f}%\n"
            f"- Consistency: {traits['consistency']*100:.0f}%\n"
        )
