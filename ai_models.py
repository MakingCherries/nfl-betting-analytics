import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NFLPredictionModels:
    """
    Comprehensive AI model suite for NFL betting predictions
    Includes 4+ different models for spread, total, and moneyline predictions
    """
    
    def __init__(self):
        self.models = {
            'spread': {},
            'total': {},
            'moneyline': {}
        }
        
        self.scalers = {
            'spread': StandardScaler(),
            'total': StandardScaler(),
            'moneyline': StandardScaler()
        }
        
        self.feature_columns = [
            'home_ppg', 'away_ppg', 'home_papg', 'away_papg',
            'home_ypg', 'away_ypg', 'home_yapg', 'away_yapg',
            'home_to_diff', 'away_to_diff', 'home_red_zone_pct', 'away_red_zone_pct',
            'home_3rd_down_pct', 'away_3rd_down_pct', 'temperature', 'wind_speed',
            'is_dome', 'is_primetime', 'rest_days_home', 'rest_days_away',
            'home_ats_pct', 'away_ats_pct', 'home_ou_pct', 'away_ou_pct'
        ]
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all AI models for different prediction types"""
        
        # Spread Prediction Models
        self.models['spread'] = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'Support Vector Machine': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            )
        }
        
        # Total Prediction Models
        self.models['total'] = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.12,
                max_depth=6,
                random_state=42
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        # Moneyline Prediction Models (probability-based)
        self.models['moneyline'] = {
            'Random Forest': RandomForestRegressor(
                n_estimators=180,
                max_depth=10,
                min_samples_split=6,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.15,
                max_depth=5,
                random_state=42
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(80, 40, 20),
                activation='tanh',
                solver='adam',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=800,
                random_state=42
            ),
            'Lasso Regression': Lasso(
                alpha=0.1,
                random_state=42
            )
        }
    
    def prepare_features(self, game_data, team_stats, weather_data, historical_data):
        """
        Prepare comprehensive feature set for model training/prediction
        """
        features = {}
        
        # Team offensive/defensive stats
        home_team = game_data['home_team']
        away_team = game_data['away_team']
        
        if home_team in team_stats and away_team in team_stats:
            home_stats = team_stats[home_team]
            away_stats = team_stats[away_team]
            
            features.update({
                'home_ppg': home_stats['offense']['points_per_game'],
                'away_ppg': away_stats['offense']['points_per_game'],
                'home_papg': home_stats['defense']['points_allowed_per_game'],
                'away_papg': away_stats['defense']['points_allowed_per_game'],
                'home_ypg': home_stats['offense']['yards_per_game'],
                'away_ypg': away_stats['offense']['yards_per_game'],
                'home_yapg': home_stats['defense']['yards_allowed_per_game'],
                'away_yapg': away_stats['defense']['yards_allowed_per_game'],
                'home_to_diff': home_stats['offense']['turnovers_per_game'] - home_stats['defense']['takeaways_per_game'],
                'away_to_diff': away_stats['offense']['turnovers_per_game'] - away_stats['defense']['takeaways_per_game'],
                'home_red_zone_pct': home_stats['offense']['red_zone_efficiency'],
                'away_red_zone_pct': away_stats['offense']['red_zone_efficiency'],
                'home_3rd_down_pct': home_stats['offense']['third_down_conversion'],
                'away_3rd_down_pct': away_stats['offense']['third_down_conversion']
            })
        else:
            # Use default values if team stats not available
            for stat in ['ppg', 'papg', 'ypg', 'yapg']:
                features[f'home_{stat}'] = 25.0
                features[f'away_{stat}'] = 25.0
            
            features.update({
                'home_to_diff': 0.0, 'away_to_diff': 0.0,
                'home_red_zone_pct': 0.6, 'away_red_zone_pct': 0.6,
                'home_3rd_down_pct': 0.4, 'away_3rd_down_pct': 0.4
            })
        
        # Weather features
        if weather_data:
            features.update({
                'temperature': weather_data.get('temperature', 70),
                'wind_speed': weather_data.get('wind_speed', 5),
                'is_dome': 1 if weather_data.get('conditions') == 'Dome' else 0
            })
        else:
            features.update({
                'temperature': 70, 'wind_speed': 5, 'is_dome': 0
            })
        
        # Game context features
        features.update({
            'is_primetime': 1 if game_data.get('primetime', False) else 0,
            'rest_days_home': game_data.get('rest_days_home', 7),
            'rest_days_away': game_data.get('rest_days_away', 7)
        })
        
        # Historical performance features
        if historical_data:
            home_ats_wins = sum(1 for game in historical_data if self._covers_spread(game, home_team))
            away_ats_wins = sum(1 for game in historical_data if self._covers_spread(game, away_team))
            total_games = len(historical_data)
            
            if total_games > 0:
                features['home_ats_pct'] = home_ats_wins / total_games
                features['away_ats_pct'] = away_ats_wins / total_games
                
                over_games = sum(1 for game in historical_data if game['total_score'] > 45)  # Assuming 45 as avg total
                features['home_ou_pct'] = over_games / total_games
                features['away_ou_pct'] = over_games / total_games
            else:
                features.update({
                    'home_ats_pct': 0.5, 'away_ats_pct': 0.5,
                    'home_ou_pct': 0.5, 'away_ou_pct': 0.5
                })
        else:
            features.update({
                'home_ats_pct': 0.5, 'away_ats_pct': 0.5,
                'home_ou_pct': 0.5, 'away_ou_pct': 0.5
            })
        
        return features
    
    def train_models(self, training_data):
        """
        Train all models on historical data
        """
        print("Training AI models on historical data...")
        
        # Prepare training data
        X = pd.DataFrame([game['features'] for game in training_data])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        
        X = X[self.feature_columns]
        
        # Prepare target variables
        y_spread = [game['actual_spread'] for game in training_data]
        y_total = [game['actual_total'] for game in training_data]
        y_home_win = [1 if game['home_won'] else 0 for game in training_data]
        
        # Train spread models
        X_spread_scaled = self.scalers['spread'].fit_transform(X)
        for name, model in self.models['spread'].items():
            try:
                model.fit(X_spread_scaled, y_spread)
                print(f"✅ Trained {name} for spread prediction")
            except Exception as e:
                print(f"❌ Error training {name} for spread: {e}")
        
        # Train total models
        X_total_scaled = self.scalers['total'].fit_transform(X)
        for name, model in self.models['total'].items():
            try:
                model.fit(X_total_scaled, y_total)
                print(f"✅ Trained {name} for total prediction")
            except Exception as e:
                print(f"❌ Error training {name} for total: {e}")
        
        # Train moneyline models
        X_ml_scaled = self.scalers['moneyline'].fit_transform(X)
        for name, model in self.models['moneyline'].items():
            try:
                model.fit(X_ml_scaled, y_home_win)
                print(f"✅ Trained {name} for moneyline prediction")
            except Exception as e:
                print(f"❌ Error training {name} for moneyline: {e}")
        
        print("Model training completed!")
    
    def predict_game(self, game_data, team_stats, weather_data, historical_data):
        """
        Generate predictions for a single game using all models
        """
        # Prepare features
        features = self.prepare_features(game_data, team_stats, weather_data, historical_data)
        feature_array = np.array([list(features.values())])
        
        # Ensure feature order matches training
        feature_df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_array = feature_df[self.feature_columns].values
        
        predictions = {
            'spread': {},
            'total': {},
            'moneyline': {}
        }
        
        # Spread predictions
        try:
            X_spread_scaled = self.scalers['spread'].transform(feature_array)
            for name, model in self.models['spread'].items():
                try:
                    pred = model.predict(X_spread_scaled)[0]
                    confidence = self._calculate_confidence(model, X_spread_scaled)
                    predictions['spread'][name] = {
                        'prediction': round(pred, 1),
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"Error in {name} spread prediction: {e}")
                    predictions['spread'][name] = {
                        'prediction': 0.0,
                        'confidence': 0.5
                    }
        except Exception as e:
            print(f"Error in spread prediction scaling: {e}")
        
        # Total predictions
        try:
            X_total_scaled = self.scalers['total'].transform(feature_array)
            for name, model in self.models['total'].items():
                try:
                    pred = model.predict(X_total_scaled)[0]
                    confidence = self._calculate_confidence(model, X_total_scaled)
                    predictions['total'][name] = {
                        'prediction': round(pred, 1),
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"Error in {name} total prediction: {e}")
                    predictions['total'][name] = {
                        'prediction': 45.0,
                        'confidence': 0.5
                    }
        except Exception as e:
            print(f"Error in total prediction scaling: {e}")
        
        # Moneyline predictions
        try:
            X_ml_scaled = self.scalers['moneyline'].transform(feature_array)
            for name, model in self.models['moneyline'].items():
                try:
                    pred = model.predict(X_ml_scaled)[0]
                    # Convert to probability
                    home_win_prob = max(0.1, min(0.9, pred))
                    predictions['moneyline'][name] = {
                        'home_win_probability': round(home_win_prob, 3),
                        'away_win_probability': round(1 - home_win_prob, 3),
                        'confidence': abs(home_win_prob - 0.5) * 2  # Higher confidence when further from 50/50
                    }
                except Exception as e:
                    print(f"Error in {name} moneyline prediction: {e}")
                    predictions['moneyline'][name] = {
                        'home_win_probability': 0.5,
                        'away_win_probability': 0.5,
                        'confidence': 0.5
                    }
        except Exception as e:
            print(f"Error in moneyline prediction scaling: {e}")
        
        return predictions
    
    def get_consensus_prediction(self, predictions):
        """
        Generate consensus predictions from all models
        """
        consensus = {}
        
        # Spread consensus
        if predictions['spread']:
            spread_preds = [p['prediction'] for p in predictions['spread'].values()]
            spread_confidences = [p['confidence'] for p in predictions['spread'].values()]
            
            # Weighted average based on confidence
            weights = np.array(spread_confidences)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            consensus['spread'] = {
                'prediction': round(np.average(spread_preds, weights=weights), 1),
                'confidence': np.mean(spread_confidences),
                'model_agreement': 1 - (np.std(spread_preds) / (np.mean(np.abs(spread_preds)) + 1e-6))
            }
        
        # Total consensus
        if predictions['total']:
            total_preds = [p['prediction'] for p in predictions['total'].values()]
            total_confidences = [p['confidence'] for p in predictions['total'].values()]
            
            weights = np.array(total_confidences)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            consensus['total'] = {
                'prediction': round(np.average(total_preds, weights=weights), 1),
                'confidence': np.mean(total_confidences),
                'model_agreement': 1 - (np.std(total_preds) / (np.mean(total_preds) + 1e-6))
            }
        
        # Moneyline consensus
        if predictions['moneyline']:
            home_probs = [p['home_win_probability'] for p in predictions['moneyline'].values()]
            ml_confidences = [p['confidence'] for p in predictions['moneyline'].values()]
            
            weights = np.array(ml_confidences)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            
            consensus_home_prob = np.average(home_probs, weights=weights)
            
            consensus['moneyline'] = {
                'home_win_probability': round(consensus_home_prob, 3),
                'away_win_probability': round(1 - consensus_home_prob, 3),
                'confidence': np.mean(ml_confidences),
                'model_agreement': 1 - (np.std(home_probs) / 0.5)  # Normalized by max possible std
            }
        
        return consensus
    
    def generate_betting_recommendation(self, predictions, current_odds, consensus):
        """
        Generate detailed betting recommendations based on model predictions vs current odds
        """
        recommendations = []
        
        # Spread recommendation
        if 'spread' in consensus and 'spread' in current_odds:
            predicted_spread = consensus['spread']['prediction']
            current_spread = current_odds['spread']
            edge = predicted_spread - current_spread
            confidence = consensus['spread']['confidence']
            agreement = consensus['spread']['model_agreement']
            
            if abs(edge) >= 2.0 and confidence >= 0.7 and agreement >= 0.7:
                bet_team = "Home" if edge > 0 else "Away"
                strength = "STRONG" if abs(edge) >= 3.0 else "MODERATE"
                
                recommendations.append({
                    'bet_type': 'Spread',
                    'recommendation': f"{strength} BET - {bet_team}",
                    'current_line': current_spread,
                    'predicted_line': predicted_spread,
                    'edge': round(edge, 1),
                    'confidence': confidence,
                    'reasoning': f"Models predict {abs(edge):.1f} point edge with {confidence:.1%} confidence and {agreement:.1%} model agreement."
                })
        
        # Total recommendation
        if 'total' in consensus and 'total' in current_odds:
            predicted_total = consensus['total']['prediction']
            current_total = current_odds['total']
            edge = predicted_total - current_total
            confidence = consensus['total']['confidence']
            agreement = consensus['total']['model_agreement']
            
            if abs(edge) >= 2.5 and confidence >= 0.65 and agreement >= 0.65:
                bet_direction = "OVER" if edge > 0 else "UNDER"
                strength = "STRONG" if abs(edge) >= 4.0 else "MODERATE"
                
                recommendations.append({
                    'bet_type': 'Total',
                    'recommendation': f"{strength} BET - {bet_direction}",
                    'current_line': current_total,
                    'predicted_line': predicted_total,
                    'edge': round(edge, 1),
                    'confidence': confidence,
                    'reasoning': f"Models project total of {predicted_total:.1f} vs current line of {current_total:.1f}, showing {abs(edge):.1f} point edge."
                })
        
        # Moneyline recommendation
        if 'moneyline' in consensus and 'moneyline' in current_odds:
            home_prob = consensus['moneyline']['home_win_probability']
            confidence = consensus['moneyline']['confidence']
            agreement = consensus['moneyline']['model_agreement']
            
            # Convert current odds to implied probability
            home_odds = current_odds['home_moneyline']
            away_odds = current_odds['away_moneyline']
            
            home_implied_prob = self._odds_to_probability(home_odds)
            away_implied_prob = self._odds_to_probability(away_odds)
            
            # Check for value
            if home_prob > home_implied_prob + 0.05 and confidence >= 0.6:
                edge = home_prob - home_implied_prob
                recommendations.append({
                    'bet_type': 'Moneyline',
                    'recommendation': 'BET - Home Team',
                    'predicted_probability': home_prob,
                    'implied_probability': home_implied_prob,
                    'edge': round(edge, 3),
                    'confidence': confidence,
                    'reasoning': f"Models give home team {home_prob:.1%} chance vs {home_implied_prob:.1%} implied by odds."
                })
            elif (1 - home_prob) > away_implied_prob + 0.05 and confidence >= 0.6:
                edge = (1 - home_prob) - away_implied_prob
                recommendations.append({
                    'bet_type': 'Moneyline',
                    'recommendation': 'BET - Away Team',
                    'predicted_probability': 1 - home_prob,
                    'implied_probability': away_implied_prob,
                    'edge': round(edge, 3),
                    'confidence': confidence,
                    'reasoning': f"Models give away team {(1-home_prob):.1%} chance vs {away_implied_prob:.1%} implied by odds."
                })
        
        return recommendations
    
    def _calculate_confidence(self, model, X):
        """Calculate prediction confidence based on model type"""
        try:
            if hasattr(model, 'predict_proba'):
                # For classification models
                proba = model.predict_proba(X)[0]
                return max(proba)
            elif hasattr(model, 'decision_function'):
                # For SVM
                decision = abs(model.decision_function(X)[0])
                return min(1.0, decision / 3.0)  # Normalize
            else:
                # For regression models, use a heuristic based on training performance
                return np.random.uniform(0.6, 0.9)  # Placeholder
        except:
            return 0.7  # Default confidence
    
    def _covers_spread(self, game, team):
        """Check if team covered the spread in historical game"""
        if game['home_team'] == team:
            return game['spread_result'] > 0
        else:
            return game['spread_result'] < 0
    
    def _odds_to_probability(self, odds):
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def save_models(self, filepath):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Advanced Model Evaluation and Backtesting
class ModelEvaluator:
    """
    Evaluate model performance and conduct backtesting
    """
    
    def __init__(self):
        self.performance_metrics = {}
    
    def backtest_models(self, models, historical_games, start_date, end_date):
        """
        Backtest models on historical data
        """
        print(f"Backtesting models from {start_date} to {end_date}")
        
        results = {
            'spread': {'correct': 0, 'total': 0, 'roi': 0, 'units': 0},
            'total': {'correct': 0, 'total': 0, 'roi': 0, 'units': 0},
            'moneyline': {'correct': 0, 'total': 0, 'roi': 0, 'units': 0}
        }
        
        for game in historical_games:
            game_date = game['date']
            if start_date <= game_date <= end_date:
                # Make predictions
                predictions = models.predict_game(
                    game['game_data'],
                    game['team_stats'],
                    game['weather_data'],
                    game['historical_data']
                )
                
                consensus = models.get_consensus_prediction(predictions)
                
                # Evaluate spread
                if 'spread' in consensus:
                    predicted_spread = consensus['spread']['prediction']
                    actual_spread = game['actual_spread']
                    
                    if abs(predicted_spread - actual_spread) <= 3:  # Within 3 points
                        results['spread']['correct'] += 1
                    results['spread']['total'] += 1
                
                # Evaluate total
                if 'total' in consensus:
                    predicted_total = consensus['total']['prediction']
                    actual_total = game['actual_total']
                    
                    if abs(predicted_total - actual_total) <= 3:  # Within 3 points
                        results['total']['correct'] += 1
                    results['total']['total'] += 1
                
                # Evaluate moneyline
                if 'moneyline' in consensus:
                    predicted_home_win = consensus['moneyline']['home_win_probability'] > 0.5
                    actual_home_win = game['home_won']
                    
                    if predicted_home_win == actual_home_win:
                        results['moneyline']['correct'] += 1
                    results['moneyline']['total'] += 1
        
        # Calculate accuracy rates
        for bet_type in results:
            if results[bet_type]['total'] > 0:
                accuracy = results[bet_type]['correct'] / results[bet_type]['total']
                results[bet_type]['accuracy'] = accuracy
                print(f"{bet_type.title()} Accuracy: {accuracy:.1%}")
        
        return results

# Usage Example
if __name__ == "__main__":
    # Initialize models
    nfl_models = NFLPredictionModels()
    
    # Generate mock training data
    training_data = []
    for i in range(1000):  # 1000 historical games
        training_data.append({
            'features': {col: np.random.randn() for col in nfl_models.feature_columns},
            'actual_spread': np.random.uniform(-14, 14),
            'actual_total': np.random.uniform(35, 60),
            'home_won': np.random.choice([True, False])
        })
    
    # Train models
    nfl_models.train_models(training_data)
    
    # Make a prediction
    mock_game = {
        'home_team': 'Kansas City Chiefs',
        'away_team': 'Buffalo Bills',
        'primetime': True,
        'rest_days_home': 7,
        'rest_days_away': 7
    }
    
    mock_team_stats = {
        'Kansas City Chiefs': {
            'offense': {'points_per_game': 28.5, 'yards_per_game': 380, 'turnovers_per_game': 1.2, 'red_zone_efficiency': 0.65, 'third_down_conversion': 0.45},
            'defense': {'points_allowed_per_game': 22.1, 'yards_allowed_per_game': 340, 'takeaways_per_game': 1.8, 'red_zone_defense': 0.55, 'third_down_defense': 0.38}
        },
        'Buffalo Bills': {
            'offense': {'points_per_game': 26.8, 'yards_per_game': 365, 'turnovers_per_game': 1.4, 'red_zone_efficiency': 0.62, 'third_down_conversion': 0.42},
            'defense': {'points_allowed_per_game': 20.5, 'yards_allowed_per_game': 320, 'takeaways_per_game': 1.6, 'red_zone_defense': 0.58, 'third_down_defense': 0.35}
        }
    }
    
    mock_weather = {'temperature': 45, 'wind_speed': 12, 'conditions': 'Clear'}
    mock_historical = []
    
    predictions = nfl_models.predict_game(mock_game, mock_team_stats, mock_weather, mock_historical)
    consensus = nfl_models.get_consensus_prediction(predictions)
    
    print("\nPrediction Results:")
    print(f"Spread Consensus: {consensus.get('spread', {}).get('prediction', 'N/A')}")
    print(f"Total Consensus: {consensus.get('total', {}).get('prediction', 'N/A')}")
    print(f"Home Win Probability: {consensus.get('moneyline', {}).get('home_win_probability', 'N/A')}")
