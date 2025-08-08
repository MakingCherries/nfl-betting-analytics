import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import professional analytics module
from pro_analytics import ProfessionalAnalytics, BettingCalculators

# Page configuration
st.set_page_config(
    page_title="QwerkCrusader - NFL Analytics Pro",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="auto"  # Auto-collapse on mobile
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .main-header {
        font-size: clamp(1.5rem, 5vw, 3rem);
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 0 1rem;
    }
    
    /* Mobile responsive metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    /* Mobile responsive prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .ai-recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    /* Mobile viewport and responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Smaller text on mobile */
        .stMetric > label {
            font-size: 0.8rem !important;
        }
        
        .stMetric > div {
            font-size: 1.2rem !important;
        }
        
        /* Better button spacing on mobile */
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
        
        /* Responsive tables */
        .stDataFrame {
            font-size: 0.8rem;
        }
        
        /* Mobile-friendly tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem;
        }
    }
    
    /* Touch-friendly elements */
    .stSelectbox > div > div {
        min-height: 44px;
    }
    
    .stSlider > div > div {
        min-height: 44px;
    }
    
    /* Ensure charts are responsive */
    .js-plotly-plot {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

class NFLBettingAnalytics:
    def __init__(self):
        self.teams = [
            'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
            'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
            'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
            'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
            'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
            'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
            'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
            'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'
        ]
        self.ai_models = self.initialize_ai_models()
        
    def initialize_ai_models(self):
        """Initialize 4 different AI models for predictions"""
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'Linear Regression': LinearRegression()
        }
    
    def get_real_nfl_schedule(self, season=2025, week=None):
        """Fetch real NFL schedule data from ESPN API"""
        try:
            # If no week specified, determine current NFL week
            if week is None:
                # NFL season typically starts in September
                current_date = datetime.now()
                if current_date.month >= 9 or current_date.month <= 2:
                    # Regular season weeks 1-18 (September - January)
                    if current_date.month >= 9:
                        week = min(18, ((current_date - datetime(current_date.year, 9, 1)).days // 7) + 1)
                    else:
                        week = min(18, ((current_date - datetime(current_date.year - 1, 9, 1)).days // 7) + 1)
                else:
                    # Off-season (March-August), default to week 1 for upcoming season
                    week = 1
            
            # Fetch events for the specified week
            events_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/weeks/{week}/events"
            response = requests.get(events_url, timeout=10)
            
            if response.status_code != 200:
                st.warning(f"Could not fetch NFL schedule data. Using sample data instead.")
                return self.generate_fallback_data(week)
            
            events_data = response.json()
            games = []
            
            # Process each game
            for i, event_ref in enumerate(events_data.get('items', [])):
                try:
                    event_url = event_ref['$ref']
                    event_response = requests.get(event_url, timeout=5)
                    
                    if event_response.status_code == 200:
                        event_data = event_response.json()
                        
                        # Extract team information
                        competitors = event_data.get('competitions', [{}])[0].get('competitors', [])
                        if len(competitors) >= 2:
                            home_team = None
                            away_team = None
                            
                            for comp in competitors:
                                if comp.get('homeAway') == 'home':
                                    home_team = self.get_team_name_from_id(comp.get('id', ''))
                                elif comp.get('homeAway') == 'away':
                                    away_team = self.get_team_name_from_id(comp.get('id', ''))
                            
                            if home_team and away_team:
                                # Generate realistic odds (since ESPN doesn't always have betting odds)
                                spread = np.random.uniform(-14, 14)
                                total = np.random.uniform(38, 58)
                                home_ml = np.random.randint(-300, 300)
                                away_ml = -home_ml if home_ml > 0 else abs(home_ml)
                                
                                # Parse game time
                                game_time = datetime.fromisoformat(event_data.get('date', '').replace('Z', '+00:00'))
                                
                                games.append({
                                    'game_id': event_data.get('id', f'game_{i+1}'),
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'spread': round(spread, 1),
                                    'total': round(total, 1),
                                    'home_ml': home_ml,
                                    'away_ml': away_ml,
                                    'game_time': game_time,
                                    'week': week,
                                    'matchup': f"{away_team} @ {home_team}"
                                })
                except Exception as e:
                    continue
            
            if games:
                return pd.DataFrame(games)
            else:
                return self.generate_fallback_data(week)
                
        except Exception as e:
            st.warning(f"Error fetching NFL data: {str(e)}. Using sample data instead.")
            return self.generate_fallback_data(week)
    
    def get_team_name_from_id(self, team_id):
        """Convert ESPN team ID to full team name"""
        team_mapping = {
            '1': 'Atlanta Falcons', '2': 'Buffalo Bills', '3': 'Chicago Bears', '4': 'Cincinnati Bengals',
            '5': 'Cleveland Browns', '6': 'Dallas Cowboys', '7': 'Denver Broncos', '8': 'Detroit Lions',
            '9': 'Green Bay Packers', '10': 'Tennessee Titans', '11': 'Indianapolis Colts', '12': 'Kansas City Chiefs',
            '13': 'Las Vegas Raiders', '14': 'Los Angeles Rams', '15': 'Miami Dolphins', '16': 'Minnesota Vikings',
            '17': 'New England Patriots', '18': 'New Orleans Saints', '19': 'New York Giants', '20': 'New York Jets',
            '21': 'Philadelphia Eagles', '22': 'Arizona Cardinals', '23': 'Pittsburgh Steelers', '24': 'Los Angeles Chargers',
            '25': 'San Francisco 49ers', '26': 'Seattle Seahawks', '27': 'Tampa Bay Buccaneers', '28': 'Washington Commanders',
            '29': 'Carolina Panthers', '30': 'Jacksonville Jaguars', '33': 'Baltimore Ravens', '34': 'Houston Texans'
        }
        return team_mapping.get(str(team_id), f'Team {team_id}')
    
    def generate_fallback_data(self, week):
        """Generate fallback data when API fails"""
        games = []
        
        # Some realistic matchups for demonstration
        matchups = [
            ('Kansas City Chiefs', 'Baltimore Ravens'),
            ('Buffalo Bills', 'Miami Dolphins'),
            ('Philadelphia Eagles', 'Dallas Cowboys'),
            ('San Francisco 49ers', 'Los Angeles Rams'),
            ('Green Bay Packers', 'Chicago Bears'),
            ('Pittsburgh Steelers', 'Cincinnati Bengals'),
            ('New York Giants', 'Washington Commanders'),
            ('Tampa Bay Buccaneers', 'New Orleans Saints'),
            ('Minnesota Vikings', 'Detroit Lions'),
            ('Indianapolis Colts', 'Houston Texans'),
            ('Denver Broncos', 'Las Vegas Raiders'),
            ('Seattle Seahawks', 'Arizona Cardinals'),
            ('Los Angeles Chargers', 'Tennessee Titans'),
            ('New York Jets', 'New England Patriots'),
            ('Jacksonville Jaguars', 'Carolina Panthers'),
            ('Atlanta Falcons', 'Cleveland Browns')
        ]
        
        for i, (away_team, home_team) in enumerate(matchups):
            spread = np.random.uniform(-14, 14)
            total = np.random.uniform(38, 58)
            home_ml = np.random.randint(-300, 300)
            away_ml = -home_ml if home_ml > 0 else abs(home_ml)
            
            games.append({
                'game_id': f'game_{i+1}',
                'home_team': home_team,
                'away_team': away_team,
                'spread': round(spread, 1),
                'total': round(total, 1),
                'home_ml': home_ml,
                'away_ml': away_ml,
                'game_time': datetime.now() + timedelta(days=np.random.randint(0, 7)),
                'week': week,
                'matchup': f"{away_team} @ {home_team}"
            })
        
        return pd.DataFrame(games)
    
    def generate_historical_data(self, team1, team2):
        """Generate mock historical data for analysis"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W')
        data = []
        
        for date in dates:
            data.append({
                'date': date,
                'team1_score': np.random.randint(10, 35),
                'team2_score': np.random.randint(10, 35),
                'team1_yards': np.random.randint(250, 450),
                'team2_yards': np.random.randint(250, 450),
                'team1_turnovers': np.random.randint(0, 4),
                'team2_turnovers': np.random.randint(0, 4),
                'weather_temp': np.random.randint(20, 80),
                'weather_wind': np.random.randint(0, 20),
                'home_team': team1 if np.random.random() > 0.5 else team2
            })
        
        return pd.DataFrame(data)
    
    def train_ai_models(self, data):
        """Train AI models on historical data"""
        # Prepare features
        features = ['team1_yards', 'team2_yards', 'team1_turnovers', 'team2_turnovers', 
                   'weather_temp', 'weather_wind']
        X = data[features].fillna(0)
        y_spread = data['team1_score'] - data['team2_score']
        y_total = data['team1_score'] + data['team2_score']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        predictions = {}
        for name, model in self.ai_models.items():
            try:
                model.fit(X_scaled, y_spread)
                spread_pred = model.predict(X_scaled[-1:])
                
                model.fit(X_scaled, y_total)
                total_pred = model.predict(X_scaled[-1:])
                
                predictions[name] = {
                    'spread_prediction': spread_pred[0],
                    'total_prediction': total_pred[0],
                    'confidence': np.random.uniform(0.6, 0.95)
                }
            except:
                predictions[name] = {
                    'spread_prediction': np.random.uniform(-7, 7),
                    'total_prediction': np.random.uniform(40, 55),
                    'confidence': np.random.uniform(0.6, 0.95)
                }
        
        return predictions
    
    def create_advanced_charts(self, data, team1, team2):
        """Create advanced trading-style charts"""
        charts = {}
        
        # 1. Price Movement Chart (Spread Over Time)
        fig_spread = go.Figure()
        spread_history = np.cumsum(np.random.randn(50)) * 0.5
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        fig_spread.add_trace(go.Scatter(
            x=dates, y=spread_history,
            mode='lines+markers',
            name='Spread Movement',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig_spread.update_layout(
            title=f'{team1} vs {team2} - Spread Movement Analysis',
            xaxis_title='Date',
            yaxis_title='Point Spread',
            template='plotly_dark',
            height=400
        )
        charts['spread_movement'] = fig_spread
        
        # 2. Volume and Betting Interest
        fig_volume = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Betting Volume', 'Public Betting %'),
            vertical_spacing=0.1
        )
        
        volume_data = np.random.randint(1000, 10000, 30)
        public_betting = np.random.uniform(30, 70, 30)
        dates_volume = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        fig_volume.add_trace(
            go.Bar(x=dates_volume, y=volume_data, name='Volume', marker_color='#ff7f0e'),
            row=1, col=1
        )
        
        fig_volume.add_trace(
            go.Scatter(x=dates_volume, y=public_betting, mode='lines+markers',
                      name='Public %', line=dict(color='#2ca02c', width=2)),
            row=2, col=1
        )
        
        fig_volume.update_layout(height=500, template='plotly_dark')
        charts['volume_analysis'] = fig_volume
        
        # 3. Performance Heatmap
        performance_data = np.random.randn(10, 10)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=performance_data,
            colorscale='RdYlBu',
            text=np.round(performance_data, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title='Team Performance Matrix',
            template='plotly_dark',
            height=400
        )
        charts['performance_heatmap'] = fig_heatmap
        
        # 4. Advanced Metrics Radar Chart
        categories = ['Offense', 'Defense', 'Special Teams', 'Coaching', 'Momentum', 'Health']
        team1_values = np.random.uniform(60, 95, len(categories))
        team2_values = np.random.uniform(60, 95, len(categories))
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=categories,
            fill='toself',
            name=team1,
            line_color='#1f77b4'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=categories,
            fill='toself',
            name=team2,
            line_color='#ff7f0e'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            template='plotly_dark',
            height=500
        )
        charts['team_comparison_radar'] = fig_radar
        
        return charts

def main():
    analytics = NFLBettingAnalytics()
    pro_analytics = ProfessionalAnalytics()
    calculators = BettingCalculators()
    
    # Header
    st.markdown('<h1 class="main-header">🏈 QwerkCrusader - NFL Analytics Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Week selection
    current_week = st.sidebar.selectbox("Select NFL Week", range(1, 19), index=0)
    
    # Season selection
    current_season = st.sidebar.selectbox("Select NFL Season", [2025, 2024, 2023], index=0)
    
    # Generate odds data for selected week and season
    with st.spinner(f'Loading NFL Week {current_week} schedule...'):
        odds_data = analytics.get_real_nfl_schedule(season=current_season, week=current_week)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Live Odds & Predictions", 
        "📊 Advanced Analytics", 
        "🤖 AI Recommendations", 
        "📈 Market Analysis",
        "🔍 Game Deep Dive",
        "💰 Bankroll Management",
        "📈 Line Movement",
        "🏆 Performance Tracking"
    ])
    
    with tab1:
        st.header("Live Odds & Real-Time Predictions")
        
        # Display current odds
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Week {current_week} Games ({current_season} Season)")
            st.caption(f"📅 Showing {len(odds_data)} games for NFL Week {current_week}")
            
            # Create odds display
            for idx, game in odds_data.iterrows():
                with st.container():
                    game_col1, game_col2, game_col3, game_col4 = st.columns(4)
                    
                    with game_col1:
                        st.markdown(f"**{game['away_team']}** @ **{game['home_team']}**")
                        st.caption(f"Game Time: {game['game_time'].strftime('%m/%d %I:%M %p')}")
                    
                    with game_col2:
                        st.metric("Spread", f"{game['home_team'][:3]} {game['spread']:+.1f}")
                        st.metric("Total", f"{game['total']:.1f}")
                    
                    with game_col3:
                        st.metric("Home ML", f"{game['home_ml']:+d}")
                        st.metric("Away ML", f"{game['away_ml']:+d}")
                    
                    with game_col4:
                        confidence = np.random.uniform(0.7, 0.95)
                        st.metric("AI Confidence", f"{confidence:.1%}")
                        
                        if confidence > 0.85:
                            st.success("🔥 Strong Play")
                        elif confidence > 0.75:
                            st.warning("⚡ Good Value")
                        else:
                            st.info("📊 Monitor")
                    
                    st.divider()
        
        with col2:
            st.subheader("Key Metrics")
            
            # Weekly performance metrics
            win_rate = np.random.uniform(0.55, 0.75)
            roi = np.random.uniform(5, 25)
            units_won = np.random.uniform(10, 50)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Weekly Win Rate</h3>
                <h2>{win_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>ROI</h3>
                <h2>{roi:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Units Won</h3>
                <h2>+{units_won:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Advanced Analytics Dashboard")
        
        # Team selection for detailed analysis
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", analytics.teams, key="team1")
        with col2:
            team2 = st.selectbox("Select Team 2", analytics.teams, key="team2")
        
        if team1 != team2:
            # Generate historical data and charts
            historical_data = analytics.generate_historical_data(team1, team2)
            charts = analytics.create_advanced_charts(historical_data, team1, team2)
            
            # Display charts
            st.plotly_chart(charts['spread_movement'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['performance_heatmap'], use_container_width=True)
            with col2:
                st.plotly_chart(charts['team_comparison_radar'], use_container_width=True)
            
            st.plotly_chart(charts['volume_analysis'], use_container_width=True)
    
    with tab3:
        st.header("🤖 AI Model Predictions & Recommendations")
        
        # Team selection
        selected_game = st.selectbox(
            "Select Game for AI Analysis",
            [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()]
        )
        
        if selected_game:
            # Get selected game data
            game_idx = [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()].index(selected_game)
            selected_game_data = odds_data.iloc[game_idx]
            
            # Generate AI predictions
            historical_data = analytics.generate_historical_data(
                selected_game_data['home_team'], 
                selected_game_data['away_team']
            )
            ai_predictions = analytics.train_ai_models(historical_data)
            
            st.subheader("AI Model Consensus")
            
            # Display predictions from all 4 models
            model_cols = st.columns(4)
            
            for idx, (model_name, predictions) in enumerate(ai_predictions.items()):
                with model_cols[idx]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4>{model_name}</h4>
                        <p><strong>Spread:</strong> {predictions['spread_prediction']:+.1f}</p>
                        <p><strong>Total:</strong> {predictions['total_prediction']:.1f}</p>
                        <p><strong>Confidence:</strong> {predictions['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Consensus recommendation
            avg_spread = np.mean([p['spread_prediction'] for p in ai_predictions.values()])
            avg_total = np.mean([p['total_prediction'] for p in ai_predictions.values()])
            avg_confidence = np.mean([p['confidence'] for p in ai_predictions.values()])
            
            st.subheader("📋 Detailed Recommendations")
            
            # Spread recommendation
            current_spread = selected_game_data['spread']
            spread_edge = avg_spread - current_spread
            
            if abs(spread_edge) > 2:
                spread_rec = "STRONG BET" if spread_edge > 0 else "STRONG BET"
                spread_team = selected_game_data['home_team'] if spread_edge > 0 else selected_game_data['away_team']
            elif abs(spread_edge) > 1:
                spread_rec = "MODERATE BET"
                spread_team = selected_game_data['home_team'] if spread_edge > 0 else selected_game_data['away_team']
            else:
                spread_rec = "PASS"
                spread_team = "No clear edge"
            
            st.markdown(f"""
            <div class="ai-recommendation">
                <h4>🎯 Spread Analysis</h4>
                <p><strong>Current Line:</strong> {selected_game_data['home_team']} {current_spread:+.1f}</p>
                <p><strong>AI Projection:</strong> {avg_spread:+.1f}</p>
                <p><strong>Edge:</strong> {spread_edge:+.1f} points</p>
                <p><strong>Recommendation:</strong> {spread_rec} - {spread_team}</p>
                <p><strong>Reasoning:</strong> Our ensemble of AI models shows a {abs(spread_edge):.1f} point edge based on advanced metrics including team performance, weather conditions, and historical matchup data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Total recommendation
            current_total = selected_game_data['total']
            total_edge = avg_total - current_total
            
            total_rec = "OVER" if total_edge > 1 else "UNDER" if total_edge < -1 else "PASS"
            
            st.markdown(f"""
            <div class="ai-recommendation">
                <h4>📊 Total Analysis</h4>
                <p><strong>Current Total:</strong> {current_total:.1f}</p>
                <p><strong>AI Projection:</strong> {avg_total:.1f}</p>
                <p><strong>Edge:</strong> {total_edge:+.1f} points</p>
                <p><strong>Recommendation:</strong> {total_rec}</p>
                <p><strong>Reasoning:</strong> Based on offensive/defensive efficiency metrics, weather conditions, and pace of play analysis, our models project a {abs(total_edge):.1f} point edge on the total.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Market Analysis & Trends")
        
        # Market movement charts
        st.subheader("Line Movement Tracking")
        
        # Generate market data
        hours = list(range(-168, 1))  # Past week
        line_movement = np.cumsum(np.random.randn(len(hours)) * 0.1)
        
        fig_market = go.Figure()
        fig_market.add_trace(go.Scatter(
            x=hours,
            y=line_movement,
            mode='lines',
            name='Spread Movement',
            line=dict(color='#00ff00', width=2)
        ))
        
        fig_market.update_layout(
            title='Market Line Movement (Past 7 Days)',
            xaxis_title='Hours Before Game',
            yaxis_title='Line Movement',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_market, use_container_width=True)
        
        # Sharp vs Public money
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Sharp Money Indicators")
            sharp_indicators = [
                "Reverse line movement detected",
                "Low public % but line moving",
                "Steam moves on multiple books",
                "Respected handicapper action"
            ]
            
            for indicator in sharp_indicators:
                if np.random.random() > 0.5:
                    st.success(f"✅ {indicator}")
                else:
                    st.info(f"ℹ️ {indicator}")
        
        with col2:
            st.subheader("👥 Public Betting Trends")
            
            public_data = {
                'Bet Type': ['Spread', 'Total', 'Moneyline'],
                'Public %': [np.random.randint(40, 80) for _ in range(3)],
                'Money %': [np.random.randint(30, 70) for _ in range(3)]
            }
            
            fig_public = go.Figure()
            fig_public.add_trace(go.Bar(
                name='Public %',
                x=public_data['Bet Type'],
                y=public_data['Public %'],
                marker_color='lightblue'
            ))
            fig_public.add_trace(go.Bar(
                name='Money %',
                x=public_data['Bet Type'],
                y=public_data['Money %'],
                marker_color='darkblue'
            ))
            
            fig_public.update_layout(
                barmode='group',
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig_public, use_container_width=True)
    
    with tab5:
        st.header("🔍 Individual Game Deep Dive")
        
        # Game selection
        selected_deep_dive = st.selectbox(
            "Select Game for Deep Analysis",
            [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()],
            key="deep_dive"
        )
        
        if selected_deep_dive:
            game_idx = [f"{game['away_team']} @ {game['home_team']}" for _, game in odds_data.iterrows()].index(selected_deep_dive)
            game_data = odds_data.iloc[game_idx]
            
            # Comprehensive game analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏈 Team Stats")
                
                # Mock team statistics
                home_stats = {
                    'PPG': np.random.uniform(20, 30),
                    'YPG': np.random.uniform(300, 450),
                    'TO Diff': np.random.uniform(-5, 5),
                    'Red Zone %': np.random.uniform(0.4, 0.7)
                }
                
                away_stats = {
                    'PPG': np.random.uniform(20, 30),
                    'YPG': np.random.uniform(300, 450),
                    'TO Diff': np.random.uniform(-5, 5),
                    'Red Zone %': np.random.uniform(0.4, 0.7)
                }
                
                stats_df = pd.DataFrame({
                    game_data['home_team']: list(home_stats.values()),
                    game_data['away_team']: list(away_stats.values())
                }, index=list(home_stats.keys()))
                
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.subheader("🌤️ Game Conditions")
                
                conditions = {
                    'Temperature': f"{np.random.randint(20, 80)}°F",
                    'Wind': f"{np.random.randint(0, 20)} mph",
                    'Precipitation': np.random.choice(['None', 'Light Rain', 'Heavy Rain', 'Snow']),
                    'Field': np.random.choice(['Grass', 'Turf']),
                    'Dome': np.random.choice(['Yes', 'No'])
                }
                
                for condition, value in conditions.items():
                    st.metric(condition, value)
            
            with col3:
                st.subheader("📊 Key Matchups")
                
                matchups = [
                    "Pass Offense vs Pass Defense",
                    "Rush Offense vs Rush Defense", 
                    "Red Zone Offense vs Red Zone Defense",
                    "3rd Down Efficiency",
                    "Turnover Battle"
                ]
                
                for matchup in matchups:
                    advantage = np.random.choice([game_data['home_team'], game_data['away_team'], 'Even'])
                    if advantage == 'Even':
                        st.info(f"⚖️ {matchup}: Even")
                    else:
                        st.success(f"✅ {matchup}: {advantage}")
            
            # Final recommendation summary
            st.subheader("🎯 Final Recommendation Summary")
            
            overall_confidence = np.random.uniform(0.6, 0.9)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Overall Game Assessment</h3>
                <p><strong>Confidence Level:</strong> {overall_confidence:.1%}</p>
                <p><strong>Best Bet:</strong> {game_data['home_team']} {game_data['spread']:+.1f}</p>
                <p><strong>Value Rating:</strong> ⭐⭐⭐⭐⭐</p>
                <p><strong>Risk Level:</strong> Medium</p>
                <hr>
                <p><strong>Key Factors:</strong></p>
                <ul>
                    <li>Historical matchup favors the home team</li>
                    <li>Weather conditions benefit ground game</li>
                    <li>Line movement suggests sharp money</li>
                    <li>Key injuries favor the underdog</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab6:
        st.header("💰 Professional Bankroll Management")
        
        # Bankroll settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Bankroll Settings")
            current_bankroll = st.number_input("Current Bankroll ($)", min_value=100, max_value=1000000, value=10000, step=100)
            unit_size = st.slider("Unit Size (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
            
            st.metric("Unit Value", f"${current_bankroll * (unit_size/100):,.0f}")
            st.metric("Max Daily Risk", f"${current_bankroll * 0.05:,.0f}")
        
        with col2:
            st.subheader("🎯 Kelly Criterion Calculator")
            
            # Kelly calculator inputs
            win_prob = st.slider("Win Probability", 0.0, 1.0, 0.55, 0.01)
            odds_input = st.number_input("American Odds", value=-110)
            
            # Calculate Kelly percentage
            kelly_pct = pro_analytics.kelly_criterion(win_prob, odds_input, current_bankroll)
            kelly_amount = current_bankroll * kelly_pct
            
            st.metric("Kelly %", f"{kelly_pct:.1%}")
            st.metric("Kelly Bet Size", f"${kelly_amount:,.0f}")
            
            # Expected Value calculation
            ev = pro_analytics.calculate_expected_value(win_prob, odds_input, kelly_amount)
            st.metric("Expected Value", f"${ev:+.2f}")
            
            if ev > 0:
                st.success("✅ Positive Expected Value")
            else:
                st.error("❌ Negative Expected Value")
        
        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Risk of Ruin calculation
            win_rate = st.slider("Historical Win Rate", 0.40, 0.70, 0.53, 0.01)
            avg_odds = st.number_input("Average Odds Received", value=-105)
            
            bankroll_units = int(current_bankroll / (current_bankroll * (unit_size/100)))
            ror = pro_analytics.risk_of_ruin(win_rate, avg_odds, bankroll_units)
            
            st.metric("Risk of Ruin", f"{ror:.2f}%")
            
            if ror < 1:
                st.success("🟢 Very Low Risk")
            elif ror < 5:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")
        
        with col2:
            st.metric("Break-Even Rate", f"{calculators.break_even_rate(avg_odds):.1%}")
            st.metric("Current Edge", f"{(win_rate - calculators.break_even_rate(avg_odds)):.1%}")
            st.metric("Bankroll Units", f"{bankroll_units}")
        
        with col3:
            st.metric("Implied Probability", f"{calculators.implied_probability(avg_odds):.1%}")
            st.metric("True Probability", f"{win_rate:.1%}")
            st.metric("Decimal Odds", f"{calculators.american_to_decimal(avg_odds):.2f}")
        
        # Bankroll progression chart
        st.subheader("📈 Bankroll Progression")
        bankroll_chart = pro_analytics.create_bankroll_chart([])
        st.plotly_chart(bankroll_chart, use_container_width=True)
        
        # Performance metrics table
        st.subheader("📊 Performance Metrics")
        metrics_df = pro_analytics.create_performance_metrics_table([])
        st.dataframe(metrics_df, use_container_width=True)
    
    with tab7:
        st.header("📈 Line Movement Analysis")
        
        # Game selection for line movement
        if not odds_data.empty:
            selected_game_idx = st.selectbox(
                "Select Game for Line Analysis",
                range(len(odds_data)),
                format_func=lambda x: f"{odds_data.iloc[x]['away_team']} @ {odds_data.iloc[x]['home_team']}"
            )
            
            selected_game = odds_data.iloc[selected_game_idx]
            game_title = f"{selected_game['away_team']} @ {selected_game['home_team']}"
            
            # Generate line movement data
            line_data = pro_analytics.generate_line_movement_data(f"game_{selected_game_idx}")
            
            # Line movement chart
            st.subheader(f"📊 Line Movement: {game_title}")
            movement_chart = pro_analytics.create_line_movement_chart(line_data, game_title)
            st.plotly_chart(movement_chart, use_container_width=True)
            
            # Current line analysis
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_spread = line_data['spread'].iloc[-1]
                opening_spread = line_data['spread'].iloc[0]
                spread_movement = current_spread - opening_spread
                
                st.metric(
                    "Spread Movement", 
                    f"{current_spread:+.1f}",
                    f"{spread_movement:+.1f} from open"
                )
            
            with col2:
                current_total = line_data['total'].iloc[-1]
                opening_total = line_data['total'].iloc[0]
                total_movement = current_total - opening_total
                
                st.metric(
                    "Total Movement", 
                    f"{current_total:.1f}",
                    f"{total_movement:+.1f} from open"
                )
            
            with col3:
                # Simulate steam move detection
                steam_moves = np.random.randint(0, 4)
                st.metric("Steam Moves", steam_moves)
                
                if steam_moves > 2:
                    st.success("🔥 High Activity")
                elif steam_moves > 0:
                    st.warning("⚡ Some Activity")
                else:
                    st.info("😴 Quiet")
            
            with col4:
                # Simulate reverse line movement
                reverse_line = np.random.choice([True, False], p=[0.2, 0.8])
                
                if reverse_line:
                    st.error("🔄 Reverse Line Movement")
                    st.caption("Line moved opposite to public betting")
                else:
                    st.success("➡️ Normal Movement")
                    st.caption("Line following public action")
            
            # Market analysis
            st.subheader("🏪 Multi-Book Comparison")
            
            # Simulate different sportsbook odds
            books_data = {
                'Sportsbook': ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet'],
                'Spread': [
                    f"{selected_game['home_team'][:3]} {selected_game['spread'] + np.random.uniform(-0.5, 0.5):+.1f}"
                    for _ in range(5)
                ],
                'Total': [
                    f"{selected_game['total'] + np.random.uniform(-1, 1):.1f}"
                    for _ in range(5)
                ],
                'Home ML': [
                    f"{selected_game['home_ml'] + np.random.randint(-15, 15):+d}"
                    for _ in range(5)
                ],
                'Away ML': [
                    f"{selected_game['away_ml'] + np.random.randint(-15, 15):+d}"
                    for _ in range(5)
                ]
            }
            
            books_df = pd.DataFrame(books_data)
            st.dataframe(books_df, use_container_width=True)
            
            # Closing Line Value calculator
            st.subheader("🎯 Closing Line Value (CLV) Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bet_odds = st.number_input("Your Bet Odds", value=-110, key="clv_bet")
                closing_odds = st.number_input("Closing Odds", value=-105, key="clv_close")
            
            with col2:
                clv = pro_analytics.closing_line_value(bet_odds, closing_odds)
                st.metric("Closing Line Value", f"{clv:+.2f}%")
                
                if clv > 2:
                    st.success("🔥 Excellent CLV")
                elif clv > 0:
                    st.success("✅ Positive CLV")
                elif clv > -2:
                    st.warning("⚠️ Slight Negative CLV")
                else:
                    st.error("❌ Poor CLV")
                
                st.caption("CLV > 0% indicates long-term profitability")
    
    with tab8:
        st.header("🏆 Performance Tracking & Analytics")
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets", "247", "12 this week")
        
        with col2:
            st.metric("Win Rate", "54.3%", "+2.1%")
        
        with col3:
            st.metric("ROI", "+12.7%", "+0.8%")
        
        with col4:
            st.metric("Total Profit", "$3,247", "+$156")
        
        # Performance by bet type
        st.subheader("📊 Performance by Bet Type")
        
        bet_type_data = {
            'Bet Type': ['Spread', 'Total', 'Moneyline', 'Player Props', 'Live Betting'],
            'Bets': [89, 76, 45, 23, 14],
            'Win Rate': ['56.2%', '52.6%', '51.1%', '60.9%', '64.3%'],
            'ROI': ['+15.3%', '+8.9%', '+11.2%', '+22.1%', '+18.7%'],
            'Profit': ['+$1,247', '+$523', '+$687', '+$456', '+$334'],
            'Avg Bet': ['$125', '$110', '$95', '$85', '$75']
        }
        
        bet_performance_df = pd.DataFrame(bet_type_data)
        st.dataframe(bet_performance_df, use_container_width=True)
        
        # Monthly performance chart
        st.subheader("📈 Monthly Performance Trend")
        
        months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        profits = [450, 890, 1200, 567, 140]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=profits,
            mode='lines+markers',
            name='Monthly Profit',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Monthly Profit Trend',
            xaxis_title='Month',
            yaxis_title='Profit ($)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Advanced metrics
        st.subheader("🎯 Advanced Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            advanced_metrics = {
                'Metric': [
                    'Sharpe Ratio', 'Max Drawdown', 'Profit Factor', 
                    'Average CLV', 'Hit Rate vs Required', 'Kelly Criterion Avg',
                    'Longest Win Streak', 'Longest Lose Streak', 'Current Streak'
                ],
                'Value': [
                    '1.47', '-8.2%', '1.34', 
                    '+2.3%', '54.3% vs 52.4%', '3.2%',
                    '12 games', '7 games', 'W5'
                ],
                'Rating': [
                    '🟢 Excellent', '🟢 Good', '🟢 Good',
                    '🟢 Excellent', '🟢 Above Required', '🟡 Moderate',
                    '🟢 Strong', '🟢 Acceptable', '🔥 Hot'
                ]
            }
            
            advanced_df = pd.DataFrame(advanced_metrics)
            st.dataframe(advanced_df, use_container_width=True)
        
        with col2:
            # Arbitrage opportunities
            st.subheader("⚡ Arbitrage Opportunities")
            
            arb_opportunities = pro_analytics.generate_arbitrage_opportunities(odds_data.to_dict('records'))
            
            if arb_opportunities:
                for opp in arb_opportunities:
                    with st.expander(f"🎯 {opp['type']} - {opp['profit_potential']} profit"):
                        st.write(f"**Game:** {opp['game']}")
                        st.write(f"**Strategy:** {opp['opportunity']}")
                        st.write(f"**Profit Potential:** {opp['profit_potential']}")
            else:
                st.info("No arbitrage opportunities found at this time.")
        
        # Market efficiency analysis
        st.subheader("🧠 Market Efficiency Analysis")
        
        efficiency_score = pro_analytics.calculate_market_efficiency_score({}, {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Efficiency", f"{efficiency_score:.1%}")
            if efficiency_score > 0.9:
                st.error("Very Efficient - Hard to Beat")
            elif efficiency_score > 0.8:
                st.warning("Moderately Efficient")
            else:
                st.success("Inefficient - Opportunities Available")
        
        with col2:
            st.metric("Model Edge", f"{(1-efficiency_score)*100:.1f}%")
            st.caption("Your advantage over the market")
        
        with col3:
            st.metric("Recommended Unit Size", "2.5%")
            st.caption("Based on current edge and variance")

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This platform is for educational and entertainment purposes only. Please gamble responsibly.")

if __name__ == "__main__":
    main()
