import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Make sure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """
    Load processed data for model training
    """
    logger.info("Loading processed data for model training")
    
    model_data_file = os.path.join(PROCESSED_DATA_DIR, "model_data.csv")
    if not os.path.exists(model_data_file):
        logger.error(f"Model data file not found: {model_data_file}")
        return None
    
    model_data = pd.read_csv(model_data_file)
    logger.info(f"Loaded model data: {len(model_data)} rows")
    
    # Load team performance data if available
    team_performance_file = os.path.join(PROCESSED_DATA_DIR, "processed_team_performance.csv")
    if os.path.exists(team_performance_file):
        team_performance = pd.read_csv(team_performance_file)
        logger.info(f"Loaded team performance data: {len(team_performance)} teams")
    else:
        team_performance = None
        logger.warning("Team performance data not found")
    
    # Load current fixtures data if available
    fixtures_file = os.path.join(PROCESSED_DATA_DIR, "processed_fixtures.csv")
    if os.path.exists(fixtures_file):
        fixtures = pd.read_csv(fixtures_file)
        logger.info(f"Loaded fixtures data: {len(fixtures)} fixtures")
    else:
        fixtures = None
        logger.warning("Fixtures data not found")
    
    data = {
        'model_data': model_data,
        'team_performance': team_performance,
        'fixtures': fixtures
    }
    
    return data

def prepare_features_targets(data):
    """
    Prepare features and target variables for model training
    """
    logger.info("Preparing features and targets for model training")
    
    if data is None:
        logger.error("No data available for feature preparation")
        return None, None, None, None
    
    # Define features to use for prediction
    # We'll use team strengths, head-to-head records, and player ratings
    feature_cols = [
        'team1_titles', 'team2_titles',
        'team1_win_percentage', 'team2_win_percentage',
        'team1_recent_form', 'team2_recent_form',
        'team1_strength', 'team2_strength'
    ]
    
    # Add head-to-head features if available
    if 'team1_h2h_win_percentage' in data.columns:
        feature_cols.extend(['team1_h2h_win_percentage', 'team2_h2h_win_percentage'])
    
    # Add player rating features if available
    player_rating_cols = [col for col in data.columns if 'rating' in col]
    feature_cols.extend(player_rating_cols)
    
    # Define target variable (team1 wins if probability > 50%)
    data['team1_wins'] = (data['team1_win_probability'] > 50).astype(int)
    
    # Create feature matrix X and target vector y
    X = data[feature_cols]
    y = data['team1_wins']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Prepared features: {X.shape[1]} features")
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    logger.info("Training Random Forest model")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting classifier
    """
    logger.info("Training Gradient Boosting model")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier
    """
    logger.info("Training XGBoost model")
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(random_state=42))
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__min_child_weight': [1, 3]
    }
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Return the best model
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on the test set
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"{model_name} accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    logger.info(f"{model_name} classification report:\n{report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a directory for model evaluation results
    eval_dir = os.path.join(MODELS_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Team 2 Wins', 'Team 1 Wins'],
                yticklabels=['Team 2 Wins', 'Team 1 Wins'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(eval_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()
    
    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def get_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from the model
    """
    logger.info(f"Extracting feature importance for {model_name}")
    
    # Get the classifier from the pipeline
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    # Extract feature importance
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning(f"Cannot extract feature importance from {model_name}")
        return None
    
    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Create a directory for feature importance results
    eval_dir = os.path.join(MODELS_DIR, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png"))
    plt.close()
    
    return feature_importance

def save_model(model, model_name):
    """
    Save trained model to disk
    """
    logger.info(f"Saving {model_name} to disk")
    
    model_file = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {model_file}")

def simulate_tournament(data, models, team_names):
    """
    Simulate the IPL tournament using the trained models
    """
    logger.info("Simulating IPL tournament")
    
    if not models:
        logger.error("No models available for tournament simulation")
        return None
    
    # Create a DataFrame to store team wins and points
    teams_df = pd.DataFrame({
        'team': team_names,
        'matches': 0,
        'wins': 0,
        'points': 0
    })
    
    # Set team as index
    teams_df.set_index('team', inplace=True)
    
    # Define the feature columns
    feature_cols = [col for col in data.columns if col not in [
        'team1', 'team2', 'team1_wins', 'team1_win_probability', 'team2_win_probability'
    ]]
    
    # Simulate league stage: each team plays against every other team twice
    for team1 in team_names:
        for team2 in team_names:
            if team1 != team2:
                # Find the matchup in the data
                match_data = data[(data['team1'] == team1) & (data['team2'] == team2)]
                
                if len(match_data) == 0:
                    logger.warning(f"No data found for matchup: {team1} vs {team2}")
                    continue
                
                # Extract features for this matchup
                features = match_data[feature_cols].values
                
                # Make predictions using all models
                predictions = []
                for model in models:
                    pred = model.predict(features)[0]
                    predictions.append(pred)
                
                # Majority vote
                team1_wins = sum(predictions) > len(models) / 2
                
                # Update team stats
                teams_df.loc[team1, 'matches'] += 1
                teams_df.loc[team2, 'matches'] += 1
                
                if team1_wins:
                    teams_df.loc[team1, 'wins'] += 1
                    teams_df.loc[team1, 'points'] += 2
                else:
                    teams_df.loc[team2, 'wins'] += 1
                    teams_df.loc[team2, 'points'] += 2
    
    # Sort teams by points (descending) and wins (descending)
    teams_df = teams_df.sort_values(['points', 'wins'], ascending=False)
    
    # Get the top 4 teams for playoffs
    playoff_teams = teams_df.head(4).index.tolist()
    
    # Simulate playoffs
    # Qualifier 1: 1st vs 2nd
    team1, team2 = playoff_teams[0], playoff_teams[1]
    q1_winner, q1_loser = simulate_match(team1, team2, data, models, feature_cols)
    
    # Eliminator: 3rd vs 4th
    team3, team4 = playoff_teams[2], playoff_teams[3]
    eliminator_winner, _ = simulate_match(team3, team4, data, models, feature_cols)
    
    # Qualifier 2: Loser of Q1 vs Winner of Eliminator
    q2_winner, _ = simulate_match(q1_loser, eliminator_winner, data, models, feature_cols)
    
    # Final: Winner of Q1 vs Winner of Q2
    champion, runner_up = simulate_match(q1_winner, q2_winner, data, models, feature_cols)
    
    # Return the results
    results = {
        'league_standings': teams_df.reset_index(),
        'playoff_teams': playoff_teams,
        'qualifier1': {'teams': [team1, team2], 'winner': q1_winner},
        'eliminator': {'teams': [team3, team4], 'winner': eliminator_winner},
        'qualifier2': {'teams': [q1_loser, eliminator_winner], 'winner': q2_winner},
        'final': {'teams': [q1_winner, q2_winner], 'winner': champion, 'runner_up': runner_up},
        'champion': champion,
        'runner_up': runner_up
    }
    
    return results

def simulate_match(team1, team2, data, models, feature_cols):
    """
    Simulate a match between two teams
    """
    # Find the matchup in the data
    match_data = data[(data['team1'] == team1) & (data['team2'] == team2)]
    
    # If we don't have this exact matchup, try the reverse
    if len(match_data) == 0:
        match_data = data[(data['team1'] == team2) & (data['team2'] == team1)]
        # Flip the result for the reverse matchup
        reverse_matchup = True
    else:
        reverse_matchup = False
    
    if len(match_data) == 0:
        logger.warning(f"No data found for matchup: {team1} vs {team2}")
        # Default to team1 as winner
        return team1, team2
    
    # Extract features for this matchup
    features = match_data[feature_cols].values
    
    # Make predictions using all models
    predictions = []
    for model in models:
        pred = model.predict(features)[0]
        predictions.append(pred)
    
    # Majority vote
    team1_wins = sum(predictions) > len(models) / 2
    
    # If it's a reverse matchup, flip the result
    if reverse_matchup:
        team1_wins = not team1_wins
    
    if team1_wins:
        return team1, team2
    else:
        return team2, team1

def analyze_results(results):
    """
    Analyze tournament simulation results
    """
    logger.info("Analyzing tournament simulation results")
    
    # Create a directory for results
    results_dir = os.path.join(MODELS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save league standings
    league_standings = results['league_standings']
    league_standings.to_csv(os.path.join(results_dir, 'league_standings.csv'), index=False)
    
    # Plot league standings
    plt.figure(figsize=(12, 8))
    sns.barplot(x='team', y='points', data=league_standings)
    plt.title('IPL 2025 Predicted League Standings')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'league_standings.png'))
    plt.close()
    
    # Create a summary report
    summary = {
        'Champion': results['champion'],
        'Runner-up': results['runner_up'],
        'Playoff Teams': results['playoff_teams'],
        'Qualifier 1 Winner': results['qualifier1']['winner'],
        'Eliminator Winner': results['eliminator']['winner'],
        'Qualifier 2 Winner': results['qualifier2']['winner']
    }
    
    # Save summary as JSON
    with open(os.path.join(results_dir, 'tournament_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Return the summary
    return summary

def predict_upcoming_fixtures(models, data, feature_cols):
    """
    Predict outcomes for upcoming fixtures
    """
    logger.info("Predicting upcoming fixtures")
    
    if 'fixtures' not in data or data['fixtures'] is None:
        logger.warning("No fixtures data available for prediction")
        # If no fixtures are available, create a sample set
        try:
            if 'model_data' in data and data['model_data'] is not None:
                model_data = data['model_data']
                teams = model_data['team1'].unique()
                
                # Create a sample fixtures dataframe for demonstration
                fixtures_data = []
                for i in range(min(10, len(teams) - 1)):  # Create up to 10 sample fixtures
                    team1 = teams[i]
                    team2 = teams[(i + 1) % len(teams)]
                    fixtures_data.append({
                        'match_id': f"sample_{i+1}",
                        'team1': team1,
                        'team2': team2,
                        'venue': 'Sample Venue',
                        'date': '2025-04-15',
                        'completed': False
                    })
                
                data['fixtures'] = pd.DataFrame(fixtures_data)
                logger.info(f"Created {len(fixtures_data)} sample fixtures for prediction")
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create sample fixtures: {str(e)}")
            return None
    
    if not models:
        logger.warning("No models available for prediction")
        return None
    
    fixtures = data['fixtures']
    model_data = data['model_data']
    
    # Get only upcoming (not completed) fixtures
    upcoming_fixtures = fixtures[fixtures['completed'] == False].copy()
    
    if len(upcoming_fixtures) == 0:
        logger.info("No upcoming fixtures to predict")
        return fixtures
    
    # Make predictions for each upcoming fixture
    for idx, fixture in upcoming_fixtures.iterrows():
        team1 = fixture['team1']
        team2 = fixture['team2']
        
        # Find the matchup in model data
        matchup = model_data[(model_data['team1'] == team1) & (model_data['team2'] == team2)]
        
        if len(matchup) == 0:
            # Try reverse matchup
            matchup = model_data[(model_data['team1'] == team2) & (model_data['team2'] == team1)]
            reverse_matchup = True
        else:
            reverse_matchup = False
        
        if len(matchup) == 0:
            logger.warning(f"No model data found for matchup: {team1} vs {team2}")
            # Create a simple data point based on team strengths if possible
            if 'team_performance' in data and data['team_performance'] is not None:
                try:
                    team_perf = data['team_performance']
                    team1_strength = team_perf[team_perf['name'] == team1]['overall_rating'].values[0]
                    team2_strength = team_perf[team_perf['name'] == team2]['overall_rating'].values[0]
                    
                    # Rough win probability based on team strengths
                    team1_win_prob = 50 + (team1_strength - team2_strength)
                    team1_win_prob = max(min(team1_win_prob, 90), 10)  # Cap between 10% and 90%
                    
                    fixtures.loc[idx, 'prediction'] = team1 if team1_win_prob > 50 else team2
                    fixtures.loc[idx, 'prediction_confidence'] = team1_win_prob if team1_win_prob > 50 else 100 - team1_win_prob
                    
                    logger.info(f"Created prediction for {team1} vs {team2} based on team strengths")
                    continue
                except Exception as e:
                    logger.error(f"Failed to create prediction based on team strengths: {str(e)}")
                    continue
            continue
        
        # Extract features for this matchup
        features = matchup[feature_cols].values
        
        # Make predictions using all models
        predictions = []
        win_probabilities = []
        
        for model in models:
            pred = model.predict(features)[0]
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0][1]  # Probability of team1 winning
                win_probabilities.append(proba if not reverse_matchup else 1 - proba)
        
        # Majority vote
        team1_wins = sum(predictions) > len(models) / 2
        
        # If it's a reverse matchup, flip the result
        if reverse_matchup:
            team1_wins = not team1_wins
        
        # Calculate average win probability
        if win_probabilities:
            avg_win_probability = sum(win_probabilities) / len(win_probabilities)
        else:
            avg_win_probability = 0.5
        
        # Update fixture with prediction
        fixtures.loc[idx, 'prediction'] = team1 if team1_wins else team2
        fixtures.loc[idx, 'prediction_confidence'] = avg_win_probability * 100 if team1_wins else (1 - avg_win_probability) * 100
    
    # Make sure the results directory exists
    results_dir = os.path.join(MODELS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions to file
    prediction_file = os.path.join(results_dir, 'fixture_predictions.csv')
    fixtures.to_csv(prediction_file, index=False)
    logger.info(f"Saved fixture predictions to {prediction_file}")
    
    return fixtures

def analyze_team_strengths(data, models):
    """
    Analyze team strengths and performances
    """
    logger.info("Analyzing team strengths")
    
    if 'team_performance' not in data or data['team_performance'] is None:
        logger.warning("No team performance data available for analysis")
        return None
    
    team_performance = data['team_performance']
    model_data = data['model_data']
    
    # Create a results directory if it doesn't exist
    results_dir = os.path.join(MODELS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate win probabilities against all other teams
    teams = team_performance['name'].unique()
    team_analysis = []
    
    for team in teams:
        # Get matchups where this team is team1
        team_matchups = model_data[model_data['team1'] == team]
        
        # Calculate average win probability against all other teams
        avg_win_prob = team_matchups['team1_win_probability'].mean()
        
        # Get team performance metrics
        team_data = team_performance[team_performance['name'] == team].iloc[0]
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Check batting strength
        if team_data['batting_avg'] > team_performance['batting_avg'].median():
            strengths.append('Batting')
        else:
            weaknesses.append('Batting')
        
        # Check bowling strength (lower economy is better)
        if team_data['bowling_economy'] < team_performance['bowling_economy'].median():
            strengths.append('Bowling')
        else:
            weaknesses.append('Bowling')
        
        # Check recent form
        if team_data['form_rating'] > team_performance['form_rating'].median():
            strengths.append('Recent Form')
        else:
            weaknesses.append('Recent Form')
        
        # Check home advantage
        if team_data['home_wins'] > team_performance['home_wins'].median():
            strengths.append('Home Performance')
        
        # Create analysis for this team
        analysis = {
            'team': team,
            'overall_rating': team_data['overall_rating'],
            'tier': team_data['tier'] if 'tier' in team_data else '',
            'avg_win_probability': avg_win_prob,
            'strengths': ', '.join(strengths),
            'weaknesses': ', '.join(weaknesses),
            'batting_avg': team_data['batting_avg'],
            'bowling_economy': team_data['bowling_economy'],
            'form_rating': team_data['form_rating']
        }
        
        team_analysis.append(analysis)
    
    # Create DataFrame
    team_analysis_df = pd.DataFrame(team_analysis)
    
    # Sort by overall rating
    team_analysis_df = team_analysis_df.sort_values('overall_rating', ascending=False)
    
    # Save team analysis
    team_analysis_file = os.path.join(results_dir, 'team_analysis.csv')
    team_analysis_df.to_csv(team_analysis_file, index=False)
    logger.info(f"Saved team analysis to {team_analysis_file}")
    
    # Create visualization of team strengths
    plt.figure(figsize=(12, 8))
    sns.barplot(x='team', y='overall_rating', data=team_analysis_df, palette='viridis')
    plt.title('Team Strength Ratings')
    plt.xlabel('Team')
    plt.ylabel('Overall Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'team_strength_ratings.png'))
    plt.close()
    
    # Create visualization of batting vs bowling
    plt.figure(figsize=(12, 8))
    plt.scatter(team_analysis_df['batting_avg'], team_analysis_df['bowling_economy'], s=100)
    
    # Add team labels to the points
    for i, row in team_analysis_df.iterrows():
        plt.annotate(row['team'], 
                     (row['batting_avg'], row['bowling_economy']),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.title('Team Batting vs Bowling Performance')
    plt.xlabel('Batting Average')
    plt.ylabel('Bowling Economy (lower is better)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'batting_vs_bowling.png'))
    plt.close()
    
    logger.info(f"Saved team analysis to {team_analysis_file}")
    
    return team_analysis_df

def create_head_to_head_analysis(data):
    """
    Create head-to-head analysis between teams
    """
    logger.info("Creating head-to-head analysis")
    
    if 'model_data' not in data or data['model_data'] is None:
        logger.warning("No model data available for head-to-head analysis")
        return None
    
    model_data = data['model_data']
    
    # Create a results directory if it doesn't exist
    results_dir = os.path.join(MODELS_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Get unique teams
    teams = sorted(model_data['team1'].unique())
    
    # Create a matrix to store head-to-head win probabilities
    h2h_matrix = pd.DataFrame(index=teams, columns=teams)
    
    # Fill the matrix with win probabilities
    for _, row in model_data.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        if 'team1_win_probability' in row:
            h2h_matrix.loc[team1, team2] = row['team1_win_probability']
            h2h_matrix.loc[team2, team1] = 100 - row['team1_win_probability']
    
    # Fill diagonal with NaN (teams don't play against themselves)
    for team in teams:
        h2h_matrix.loc[team, team] = np.nan
    
    # Fill any missing values with 50 (even probability)
    h2h_matrix = h2h_matrix.fillna(50.0)
    
    # Save head-to-head matrix
    h2h_file = os.path.join(results_dir, 'head_to_head_matrix.csv')
    h2h_matrix.to_csv(h2h_file)
    
    # Create a heatmap visualization
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(h2h_matrix, dtype=bool))  # Only show lower triangle
    heatmap = sns.heatmap(h2h_matrix, annot=True, fmt='.1f', cmap='YlGnBu', 
                         mask=mask, vmin=0, vmax=100)
    plt.title('Head-to-Head Win Probability Matrix (%)')
    plt.xlabel('Team (as Opponent)')
    plt.ylabel('Team')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'head_to_head_heatmap.png'))
    plt.close()
    
    logger.info(f"Saved head-to-head analysis to {h2h_file}")
    
    return h2h_matrix

def main():
    """
    Main function for model training and evaluation
    """
    logger.info("Starting model training and evaluation")
    
    try:
        # Load data
        data = load_data()
        
        if data is None or 'model_data' not in data or data['model_data'] is None:
            logger.error("No data available for training")
            return
        
        # For convenience, use model_data directly for training
        model_data = data['model_data']
        
        # Prepare features and targets
        X_train, X_test, y_train, y_test, feature_cols = prepare_features_targets(model_data)
        
        if X_train is None:
            logger.error("Could not prepare features and targets")
            return
        
        # Create evaluation directory
        eval_dir = os.path.join(MODELS_DIR, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create results directory
        results_dir = os.path.join(MODELS_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Train models
        models = []
        try:
            # Train Random Forest model
            rf_model = train_random_forest(X_train, y_train)
            rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
            rf_importance = get_feature_importance(rf_model, feature_cols, "Random Forest")
            save_model(rf_model, "random_forest")
            models.append(rf_model)
            logger.info("Random Forest model trained successfully")
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
        
        try:
            # Train Gradient Boosting model
            gb_model = train_gradient_boosting(X_train, y_train)
            gb_metrics = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
            gb_importance = get_feature_importance(gb_model, feature_cols, "Gradient Boosting")
            save_model(gb_model, "gradient_boosting")
            models.append(gb_model)
            logger.info("Gradient Boosting model trained successfully")
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {str(e)}")
        
        try:
            # Train XGBoost model
            xgb_model = train_xgboost(X_train, y_train)
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            xgb_importance = get_feature_importance(xgb_model, feature_cols, "XGBoost")
            save_model(xgb_model, "xgboost")
            models.append(xgb_model)
            logger.info("XGBoost model trained successfully")
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
        
        if not models:
            logger.error("No models were trained successfully")
            return
        
        # Compare models
        models_comparison = pd.DataFrame({
            'Model': ["Random Forest", "Gradient Boosting", "XGBoost"],
            'Accuracy': [
                rf_metrics['accuracy'] if 'rf_metrics' in locals() else 0,
                gb_metrics['accuracy'] if 'gb_metrics' in locals() else 0,
                xgb_metrics['accuracy'] if 'xgb_metrics' in locals() else 0
            ]
        })
        
        logger.info("Model comparison:")
        logger.info(models_comparison)
        
        # Plot model comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', data=models_comparison)
        plt.title('Model Comparison - Accuracy')
        plt.ylim(0.5, 1.0)
        plt.savefig(os.path.join(eval_dir, 'model_comparison.png'))
        plt.close()
        
        # Predict upcoming fixtures
        predicted_fixtures = predict_upcoming_fixtures(models, data, feature_cols)
        
        # Analyze team strengths
        team_analysis = analyze_team_strengths(data, models)
        
        # Create head-to-head analysis
        h2h_analysis = create_head_to_head_analysis(data)
        
        # Simulate the IPL tournament
        try:
            # Get unique team names from the model data
            model_data = data['model_data']
            team_names = model_data['team1'].unique()
            
            logger.info(f"Simulating tournament with {len(team_names)} teams")
            tournament_results = simulate_tournament(model_data, models, team_names)
            
            if tournament_results:
                summary = analyze_results(tournament_results)
                logger.info("Tournament simulation summary:")
                for key, value in summary.items():
                    logger.info(f"{key}: {value}")
                
                # Add team analysis to tournament summary
                if team_analysis is not None:
                    # Create a comprehensive analysis JSON
                    comprehensive_analysis = {
                        'tournament_summary': summary,
                        'team_analysis': team_analysis.to_dict(orient='records'),
                        'upcoming_fixtures': predicted_fixtures[predicted_fixtures['completed'] == False].to_dict(orient='records') if predicted_fixtures is not None else []
                    }
                    
                    # Save comprehensive analysis
                    with open(os.path.join(results_dir, 'comprehensive_analysis.json'), 'w') as f:
                        json.dump(comprehensive_analysis, f, indent=4)
            else:
                logger.error("Tournament simulation failed")
        except Exception as e:
            logger.error(f"Error during tournament simulation: {str(e)}")
        
        logger.info("Model training and evaluation completed")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 