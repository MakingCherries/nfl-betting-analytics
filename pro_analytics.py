"""
Professional Gambling Analytics Module
Advanced features for serious NFL bettors including bankroll management,
line movement tracking, and closing line value analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Tuple, Optional
import math

class ProfessionalAnalytics:
    """
    Professional-grade analytics for serious NFL bettors
    """
    
    def __init__(self):
        self.bet_history = []
        self.bankroll_history = []
        self.line_movements = {}
        
    def kelly_criterion(self, win_probability: float, odds: float, bankroll: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        
        Args:
            win_probability: Probability of winning (0-1)
            odds: Decimal odds (e.g., 1.91 for -110)
            bankroll: Current bankroll
            
        Returns:
            Optimal bet size as percentage of bankroll
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0
            
        # Convert to decimal odds if needed
        if odds < 0:
            decimal_odds = (100 / abs(odds)) + 1
        else:
            decimal_odds = (odds / 100) + 1
            
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = win probability, q = lose probability
        b = decimal_odds - 1
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 25% of bankroll for safety
        return min(max(kelly_fraction, 0), 0.25)
    
    def calculate_expected_value(self, win_prob: float, odds: float, bet_amount: float) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            win_prob: Probability of winning
            odds: American odds
            bet_amount: Bet size
            
        Returns:
            Expected value in dollars
        """
        if odds < 0:
            win_amount = bet_amount * (100 / abs(odds))
        else:
            win_amount = bet_amount * (odds / 100)
            
        ev = (win_prob * win_amount) - ((1 - win_prob) * bet_amount)
        return ev
    
    def closing_line_value(self, bet_odds: float, closing_odds: float) -> float:
        """
        Calculate Closing Line Value (CLV) - key metric for long-term success
        
        Args:
            bet_odds: Odds when bet was placed
            closing_odds: Closing odds
            
        Returns:
            CLV as percentage
        """
        def odds_to_probability(odds):
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            else:
                return 100 / (odds + 100)
        
        bet_prob = odds_to_probability(bet_odds)
        closing_prob = odds_to_probability(closing_odds)
        
        clv = (closing_prob - bet_prob) / bet_prob * 100
        return clv
    
    def risk_of_ruin(self, win_rate: float, avg_odds: float, bankroll_units: int) -> float:
        """
        Calculate probability of losing entire bankroll
        
        Args:
            win_rate: Historical win percentage
            avg_odds: Average odds received
            bankroll_units: Bankroll in betting units
            
        Returns:
            Risk of ruin percentage
        """
        if win_rate >= 1 or win_rate <= 0:
            return 0 if win_rate >= 1 else 100
            
        # Convert odds to decimal
        if avg_odds < 0:
            decimal_odds = (100 / abs(avg_odds)) + 1
        else:
            decimal_odds = (avg_odds / 100) + 1
            
        # Calculate advantage
        advantage = (win_rate * decimal_odds) - 1
        
        if advantage <= 0:
            return 100  # No edge = certain ruin
            
        # Risk of ruin formula for advantageous betting
        q_over_p = (1 - win_rate) / win_rate
        payout_ratio = decimal_odds - 1
        
        if q_over_p == 1 / payout_ratio:
            return 100 / (bankroll_units + 1)
        else:
            ratio = q_over_p / (1 / payout_ratio)
            return (ratio ** bankroll_units) * 100
    
    def generate_line_movement_data(self, game_id: str, days_before: int = 7) -> pd.DataFrame:
        """
        Generate realistic line movement data for visualization
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days_before),
            end=datetime.now(),
            freq='H'
        )
        
        # Simulate line movement with realistic patterns
        initial_spread = np.random.uniform(-14, 14)
        spreads = [initial_spread]
        
        for i in range(1, len(dates)):
            # Add some randomness with trend
            change = np.random.normal(0, 0.5)
            # Add steam moves (sharp money)
            if np.random.random() < 0.05:  # 5% chance of steam move
                change += np.random.choice([-2, 2]) * np.random.random()
            
            new_spread = spreads[-1] + change
            spreads.append(new_spread)
        
        # Generate corresponding totals
        initial_total = np.random.uniform(38, 58)
        totals = [initial_total]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.3)
            if np.random.random() < 0.03:  # Steam moves less common for totals
                change += np.random.choice([-1.5, 1.5]) * np.random.random()
            
            new_total = max(totals[-1] + change, 30)  # Keep reasonable
            totals.append(new_total)
        
        return pd.DataFrame({
            'datetime': dates,
            'spread': spreads,
            'total': totals,
            'volume': np.random.exponential(1000, len(dates))  # Betting volume
        })
    
    def create_line_movement_chart(self, game_data: pd.DataFrame, game_title: str) -> go.Figure:
        """
        Create professional line movement visualization
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Spread Movement', 'Total Movement'],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Spread movement
        fig.add_trace(
            go.Scatter(
                x=game_data['datetime'],
                y=game_data['spread'],
                mode='lines+markers',
                name='Spread',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=4),
                hovertemplate='<b>%{y:.1f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Total movement
        fig.add_trace(
            go.Scatter(
                x=game_data['datetime'],
                y=game_data['total'],
                mode='lines+markers',
                name='Total',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=4),
                hovertemplate='<b>%{y:.1f}</b><br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=game_data['datetime'],
                y=game_data['volume'],
                name='Volume',
                opacity=0.3,
                marker_color='gray',
                yaxis='y3',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            title=f'Line Movement Analysis - {game_title}',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Spread", row=1, col=1)
        fig.update_yaxes(title_text="Total", row=2, col=1)
        
        return fig
    
    def create_bankroll_chart(self, bet_history: List[Dict]) -> go.Figure:
        """
        Create bankroll progression chart with drawdown analysis
        """
        if not bet_history:
            # Generate sample data
            dates = pd.date_range(start='2024-09-01', end='2024-12-31', freq='D')
            bankroll = [10000]  # Starting bankroll
            
            for i in range(1, len(dates)):
                # Simulate realistic betting results
                if np.random.random() < 0.53:  # 53% win rate
                    change = np.random.uniform(50, 300)  # Win
                else:
                    change = -np.random.uniform(100, 250)  # Loss
                
                new_bankroll = max(bankroll[-1] + change, 0)
                bankroll.append(new_bankroll)
            
            bet_history = [{'date': date, 'bankroll': br} for date, br in zip(dates, bankroll)]
        
        df = pd.DataFrame(bet_history)
        
        # Calculate drawdown
        peak = df['bankroll'].expanding().max()
        drawdown = (df['bankroll'] - peak) / peak * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Bankroll Progression', 'Drawdown Analysis'],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Bankroll progression
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bankroll'],
                mode='lines',
                name='Bankroll',
                line=dict(color='green', width=3),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)'
            ),
            row=1, col=1
        )
        
        # Peak line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=peak,
                mode='lines',
                name='Peak',
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=drawdown,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Bankroll Management Analysis',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Bankroll ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_performance_metrics_table(self, bet_history: List[Dict]) -> pd.DataFrame:
        """
        Create comprehensive performance metrics table
        """
        if not bet_history:
            # Generate sample performance data
            return pd.DataFrame({
                'Metric': [
                    'Total Bets', 'Win Rate', 'ROI', 'Total Profit/Loss',
                    'Average Bet Size', 'Largest Win', 'Largest Loss',
                    'Current Streak', 'Longest Win Streak', 'Longest Lose Streak',
                    'Closing Line Value', 'Sharpe Ratio', 'Max Drawdown',
                    'Profit Factor', 'Kelly Criterion Avg'
                ],
                'Value': [
                    '247', '54.3%', '+12.7%', '+$3,247',
                    '$125', '+$450', '-$275',
                    'W5', '12', '7',
                    '+2.3%', '1.47', '-8.2%',
                    '1.34', '3.2%'
                ]
            })
        
        # Calculate real metrics from bet history
        df = pd.DataFrame(bet_history)
        total_bets = len(df)
        wins = sum(1 for bet in bet_history if bet.get('result') == 'win')
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_profit = sum(bet.get('profit', 0) for bet in bet_history)
        total_wagered = sum(bet.get('amount', 0) for bet in bet_history)
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        return pd.DataFrame({
            'Metric': ['Total Bets', 'Win Rate', 'ROI', 'Total Profit/Loss'],
            'Value': [
                str(total_bets),
                f"{win_rate:.1%}",
                f"{roi:+.1f}%",
                f"${total_profit:+,.0f}"
            ]
        })
    
    def calculate_market_efficiency_score(self, predictions: Dict, actual_results: Dict) -> float:
        """
        Calculate how efficient the betting market is based on model accuracy
        """
        if not predictions or not actual_results:
            return 0.85  # Assume 85% market efficiency
        
        # Compare predictions to actual results
        accuracy_scores = []
        for game_id in predictions:
            if game_id in actual_results:
                pred = predictions[game_id]
                actual = actual_results[game_id]
                
                # Calculate accuracy for spread predictions
                spread_error = abs(pred.get('spread', 0) - actual.get('final_margin', 0))
                spread_accuracy = max(0, 1 - (spread_error / 14))  # Normalize by typical spread range
                accuracy_scores.append(spread_accuracy)
        
        if accuracy_scores:
            model_accuracy = np.mean(accuracy_scores)
            # Market efficiency = 1 - (model_edge)
            return min(0.95, 0.7 + (0.25 * model_accuracy))
        
        return 0.85
    
    def generate_arbitrage_opportunities(self, games: List[Dict]) -> List[Dict]:
        """
        Identify potential arbitrage opportunities across different bet types
        """
        opportunities = []
        
        for game in games:
            # Simulate different sportsbook odds
            books = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars']
            
            # Check for spread arbitrage
            spreads = {}
            for book in books:
                base_spread = game.get('spread', -3)
                spreads[book] = {
                    'spread': base_spread + np.random.uniform(-0.5, 0.5),
                    'odds': -110 + np.random.randint(-15, 15)
                }
            
            # Look for arbitrage (simplified)
            spread_values = [s['spread'] for s in spreads.values()]
            if max(spread_values) - min(spread_values) > 1.0:  # Significant difference
                opportunities.append({
                    'game': f"{game.get('away_team', 'Team A')} @ {game.get('home_team', 'Team B')}",
                    'type': 'Spread Arbitrage',
                    'opportunity': f"Bet {min(spreads, key=lambda x: spreads[x]['spread'])} and {max(spreads, key=lambda x: spreads[x]['spread'])}",
                    'profit_potential': f"{np.random.uniform(0.5, 3.0):.1f}%"
                })
        
        return opportunities[:5]  # Return top 5 opportunities

class BettingCalculators:
    """
    Collection of betting calculators and utilities
    """
    
    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> float:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)
    
    @staticmethod
    def implied_probability(american_odds: float) -> float:
        """Calculate implied probability from American odds"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    @staticmethod
    def break_even_rate(american_odds: float) -> float:
        """Calculate break-even win rate needed"""
        return BettingCalculators.implied_probability(american_odds)
    
    @staticmethod
    def units_to_win(american_odds: float, bet_amount: float) -> float:
        """Calculate units to win from bet"""
        if american_odds > 0:
            return bet_amount * (american_odds / 100)
        else:
            return bet_amount * (100 / abs(american_odds))
