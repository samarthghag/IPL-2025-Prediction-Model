import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import zipfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Make sure the processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def extract_matches_data():
    """
    Extract matches data from zip file if needed
    """
    zip_file = os.path.join(RAW_DATA_DIR, "ipl_csv.zip")
    matches_dir = os.path.join(RAW_DATA_DIR, "ipl_csv")
    
    if os.path.exists(zip_file) and not os.path.exists(matches_dir):
        logger.info("Extracting matches data from zip file")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info("Matches data extracted successfully")
    elif os.path.exists(matches_dir):
        logger.info("Matches data already extracted")
    else:
        logger.warning("No matches data found")

def load_data():
    """
    Load all raw data files
    """
    logger.info("Loading raw data")
    
    # Extract matches data if needed
    extract_matches_data()
    
    data = {}
    
    # Load teams data
    teams_file = os.path.join(RAW_DATA_DIR, "ipl_teams.csv")
    if os.path.exists(teams_file):
        data['teams'] = pd.read_csv(teams_file)
        logger.info(f"Loaded teams data: {len(data['teams'])} teams")
    else:
        logger.warning(f"Teams data file not found: {teams_file}")
    
    # Load players data
    players_file = os.path.join(RAW_DATA_DIR, "ipl_players.csv")
    if os.path.exists(players_file):
        data['players'] = pd.read_csv(players_file)
        logger.info(f"Loaded players data: {len(data['players'])} players")
    else:
        logger.warning(f"Players data file not found: {players_file}")
    
    # Load venues data
    venues_file = os.path.join(RAW_DATA_DIR, "ipl_venues.csv")
    if os.path.exists(venues_file):
        data['venues'] = pd.read_csv(venues_file)
        logger.info(f"Loaded venues data: {len(data['venues'])} venues")
    else:
        logger.warning(f"Venues data file not found: {venues_file}")
    
    # Load seasons data
    seasons_file = os.path.join(RAW_DATA_DIR, "ipl_seasons.csv")
    if os.path.exists(seasons_file):
        data['seasons'] = pd.read_csv(seasons_file)
        logger.info(f"Loaded seasons data: {len(data['seasons'])} seasons")
    else:
        logger.warning(f"Seasons data file not found: {seasons_file}")
    
    # Load current fixtures data
    fixtures_file = os.path.join(RAW_DATA_DIR, "current_fixtures.csv")
    if os.path.exists(fixtures_file):
        data['current_fixtures'] = pd.read_csv(fixtures_file)
        logger.info(f"Loaded current fixtures data: {len(data['current_fixtures'])} fixtures")
    else:
        logger.warning(f"Current fixtures data file not found: {fixtures_file}")
    
    # Load team performance data
    team_performance_file = os.path.join(RAW_DATA_DIR, "team_performance.csv")
    if os.path.exists(team_performance_file):
        data['team_performance'] = pd.read_csv(team_performance_file)
        logger.info(f"Loaded team performance data: {len(data['team_performance'])} teams")
    else:
        logger.warning(f"Team performance data file not found: {team_performance_file}")
    
    # Try to load match data if available
    matches_dir = os.path.join(RAW_DATA_DIR, "ipl_csv")
    if os.path.exists(matches_dir):
        try:
            # Combine all match CSV files into a single DataFrame
            matches_list = []
            for file in os.listdir(matches_dir):
                if file.endswith(".csv"):
                    match_file = os.path.join(matches_dir, file)
                    match_df = pd.read_csv(match_file)
                    matches_list.append(match_df)
            
            if matches_list:
                data['matches'] = pd.concat(matches_list, ignore_index=True)
                logger.info(f"Loaded matches data: {len(data['matches'])} matches")
            else:
                logger.warning("No match files found in the matches directory")
                data['matches'] = create_mock_matches_data()
        except Exception as e:
            logger.error(f"Error loading matches data: {str(e)}")
            data['matches'] = create_mock_matches_data()
    else:
        logger.warning(f"Matches directory not found: {matches_dir}")
        # Create mock matches data if real data isn't available
        data['matches'] = create_mock_matches_data()
        
    return data

def create_mock_matches_data():
    """
    Create mock matches data if real data isn't available
    """
    logger.info("Creating mock matches data")
    
    # Get teams
    teams_file = os.path.join(RAW_DATA_DIR, "ipl_teams.csv")
    if os.path.exists(teams_file):
        teams = pd.read_csv(teams_file)['name'].tolist()
    else:
        teams = [
            "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
            "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
            "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
            "Sunrisers Hyderabad"
        ]
    
    # Get venues
    venues_file = os.path.join(RAW_DATA_DIR, "ipl_venues.csv")
    if os.path.exists(venues_file):
        venues = pd.read_csv(venues_file)['name'].tolist()
    else:
        venues = [
            "Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens", 
            "Narendra Modi Stadium", "Sawai Mansingh Stadium", "Rajiv Gandhi International Stadium",
            "M. Chinnaswamy Stadium", "Arun Jaitley Stadium", "PCA Stadium, Mohali", 
            "BRSABV Ekana Cricket Stadium"
        ]
    
    # Create mock matches data for the last 5 seasons
    import random
    
    matches = []
    
    # Seasons from 2020 to 2024
    for year in range(2020, 2025):
        # Regular season: each team plays against each other team twice
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j:  # Teams don't play against themselves
                    # Match 1
                    venue = venues[i % len(venues)]  # Home venue for team1
                    
                    # Simulate scores
                    team1_score = random.randint(120, 220)
                    team2_score = random.randint(120, 220)
                    
                    winner = team1 if team1_score > team2_score else team2
                    margin = abs(team1_score - team2_score)
                    
                    match = {
                        'id': len(matches) + 1,
                        'season': year,
                        'date': f"{year}-04-{random.randint(1, 30):02d}",  # Random day in April
                        'venue': venue,
                        'team1': team1,
                        'team2': team2,
                        'toss_winner': random.choice([team1, team2]),
                        'toss_decision': random.choice(['bat', 'field']),
                        'winner': winner,
                        'margin': margin,
                        'method': 'normal',
                        'player_of_match': '',  # Would be filled with actual data
                        'team1_score': team1_score,
                        'team2_score': team2_score
                    }
                    
                    matches.append(match)
                    
                    # Match 2 (return leg)
                    venue = venues[j % len(venues)]  # Home venue for team2
                    
                    # Simulate scores
                    team1_score = random.randint(120, 220)
                    team2_score = random.randint(120, 220)
                    
                    winner = team1 if team1_score > team2_score else team2
                    margin = abs(team1_score - team2_score)
                    
                    match = {
                        'id': len(matches) + 1,
                        'season': year,
                        'date': f"{year}-05-{random.randint(1, 20):02d}",  # Random day in May
                        'venue': venue,
                        'team1': team1,
                        'team2': team2,
                        'toss_winner': random.choice([team1, team2]),
                        'toss_decision': random.choice(['bat', 'field']),
                        'winner': winner,
                        'margin': margin,
                        'method': 'normal',
                        'player_of_match': '',  # Would be filled with actual data
                        'team1_score': team1_score,
                        'team2_score': team2_score
                    }
                    
                    matches.append(match)
        
        # Playoffs: top 4 teams
        # For simplicity, we'll just pick 4 random teams for playoffs
        playoff_teams = random.sample(teams, 4)
        
        # Qualifier 1: 1st vs 2nd
        team1, team2 = playoff_teams[0], playoff_teams[1]
        team1_score = random.randint(150, 220)
        team2_score = random.randint(140, 210)
        winner = team1 if team1_score > team2_score else team2
        
        match = {
            'id': len(matches) + 1,
            'season': year,
            'date': f"{year}-05-{random.randint(20, 25):02d}",
            'venue': random.choice(venues),
            'team1': team1,
            'team2': team2,
            'toss_winner': random.choice([team1, team2]),
            'toss_decision': random.choice(['bat', 'field']),
            'winner': winner,
            'margin': abs(team1_score - team2_score),
            'method': 'normal',
            'player_of_match': '',
            'team1_score': team1_score,
            'team2_score': team2_score,
            'match_type': 'Qualifier 1'
        }
        
        matches.append(match)
        
        # Eliminator: 3rd vs 4th
        team3, team4 = playoff_teams[2], playoff_teams[3]
        team3_score = random.randint(150, 220)
        team4_score = random.randint(140, 210)
        eliminator_winner = team3 if team3_score > team4_score else team4
        
        match = {
            'id': len(matches) + 1,
            'season': year,
            'date': f"{year}-05-{random.randint(21, 26):02d}",
            'venue': random.choice(venues),
            'team1': team3,
            'team2': team4,
            'toss_winner': random.choice([team3, team4]),
            'toss_decision': random.choice(['bat', 'field']),
            'winner': eliminator_winner,
            'margin': abs(team3_score - team4_score),
            'method': 'normal',
            'player_of_match': '',
            'team1_score': team3_score,
            'team2_score': team4_score,
            'match_type': 'Eliminator'
        }
        
        matches.append(match)
        
        # Qualifier 2: Loser of Q1 vs Winner of Eliminator
        q1_loser = team2 if winner == team1 else team1
        q2_team1, q2_team2 = q1_loser, eliminator_winner
        q2_team1_score = random.randint(150, 220)
        q2_team2_score = random.randint(140, 210)
        q2_winner = q2_team1 if q2_team1_score > q2_team2_score else q2_team2
        
        match = {
            'id': len(matches) + 1,
            'season': year,
            'date': f"{year}-05-{random.randint(23, 27):02d}",
            'venue': random.choice(venues),
            'team1': q2_team1,
            'team2': q2_team2,
            'toss_winner': random.choice([q2_team1, q2_team2]),
            'toss_decision': random.choice(['bat', 'field']),
            'winner': q2_winner,
            'margin': abs(q2_team1_score - q2_team2_score),
            'method': 'normal',
            'player_of_match': '',
            'team1_score': q2_team1_score,
            'team2_score': q2_team2_score,
            'match_type': 'Qualifier 2'
        }
        
        matches.append(match)
        
        # Final: Winner of Q1 vs Winner of Q2
        final_team1, final_team2 = winner, q2_winner
        final_team1_score = random.randint(160, 230)
        final_team2_score = random.randint(150, 220)
        final_winner = final_team1 if final_team1_score > final_team2_score else final_team2
        
        match = {
            'id': len(matches) + 1,
            'season': year,
            'date': f"{year}-05-{random.randint(28, 31):02d}",
            'venue': random.choice(venues),
            'team1': final_team1,
            'team2': final_team2,
            'toss_winner': random.choice([final_team1, final_team2]),
            'toss_decision': random.choice(['bat', 'field']),
            'winner': final_winner,
            'margin': abs(final_team1_score - final_team2_score),
            'method': 'normal',
            'player_of_match': '',
            'team1_score': final_team1_score,
            'team2_score': final_team2_score,
            'match_type': 'Final'
        }
        
        matches.append(match)
    
    return pd.DataFrame(matches)

def process_teams_data(data):
    """
    Process teams data
    """
    logger.info("Processing teams data")
    
    if 'teams' not in data:
        logger.warning("Teams data not available")
        return None
    
    teams_df = data['teams'].copy()
    
    # Calculate win percentage from matches data
    if 'matches' in data:
        matches_df = data['matches']
        
        # Initialize win count and total matches for each team
        teams_stats = {team: {'wins': 0, 'matches': 0} for team in teams_df['name']}
        
        # Count wins and matches for each team
        for _, match in matches_df.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            winner = match.get('winner')
            
            if team1 in teams_stats:
                teams_stats[team1]['matches'] += 1
                if winner == team1:
                    teams_stats[team1]['wins'] += 1
            
            if team2 in teams_stats:
                teams_stats[team2]['matches'] += 1
                if winner == team2:
                    teams_stats[team2]['wins'] += 1
        
        # Calculate win percentage
        win_percentage = {}
        for team, stats in teams_stats.items():
            if stats['matches'] > 0:
                win_percentage[team] = round(stats['wins'] / stats['matches'] * 100, 2)
            else:
                win_percentage[team] = 0
        
        # Add win percentage to teams DataFrame
        teams_df['win_percentage'] = teams_df['name'].map(win_percentage)
    
    # Calculate recent form (last 10 matches) if matches data is available
    if 'matches' in data:
        matches_df = data['matches'].sort_values('date')
        
        # Function to get recent form for a team
        def get_recent_form(team_name, n=10):
            team_matches = matches_df[(matches_df['team1'] == team_name) | (matches_df['team2'] == team_name)]
            
            if len(team_matches) == 0:
                return 0
            
            # Get the last n matches
            recent_matches = team_matches.tail(n)
            
            wins = sum(1 for _, match in recent_matches.iterrows() 
                     if match['winner'] == team_name)
            
            return wins / len(recent_matches) * 100
        
        # Add recent form to teams DataFrame
        teams_df['recent_form'] = teams_df['name'].apply(lambda x: get_recent_form(x))
    
    # Calculate team strength based on titles and win percentage
    if 'titles' in teams_df.columns and 'win_percentage' in teams_df.columns:
        # Normalize titles and win percentage (min-max scaling)
        max_titles = teams_df['titles'].max() if teams_df['titles'].max() > 0 else 1
        teams_df['normalized_titles'] = teams_df['titles'] / max_titles
        
        max_win_pct = teams_df['win_percentage'].max() if teams_df['win_percentage'].max() > 0 else 1
        teams_df['normalized_win_pct'] = teams_df['win_percentage'] / max_win_pct
        
        # Calculate team strength (weighted average)
        teams_df['team_strength'] = 0.5 * teams_df['normalized_titles'] + 0.5 * teams_df['normalized_win_pct']
        
        # Scale to 0-100
        teams_df['team_strength'] = teams_df['team_strength'] * 100
    
    # Save processed teams data
    teams_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_teams.csv"), index=False)
    logger.info(f"Saved processed teams data: {len(teams_df)} teams")
    
    return teams_df

def process_players_data(data):
    """
    Process players data
    """
    logger.info("Processing players data")
    
    if 'players' not in data:
        logger.warning("Players data not available")
        return None
    
    players_df = data['players'].copy()
    
    # Calculate player rating based on stats
    def calculate_player_rating(row):
        role = row['role']
        rating = 0
        
        if role == 'Batsman' or role == 'Wicket-keeper':
            # For batsmen, higher batting average and strike rate is better
            avg_weight = 0.6
            sr_weight = 0.4
            
            # Normalize batting stats (assuming max batting_avg of 60 and max strike_rate of 200)
            norm_avg = min(row['batting_avg'] / 60, 1) if row['batting_avg'] > 0 else 0
            norm_sr = min(row['strike_rate'] / 200, 1) if row['strike_rate'] > 0 else 0
            
            rating = avg_weight * norm_avg + sr_weight * norm_sr
            
        elif role == 'Bowler':
            # For bowlers, lower economy rate is better
            # Normalize bowling economy (assuming best is 6.0 and worst is 12.0)
            if row['bowling_economy'] > 0:
                # Invert the scale: lower economy is better
                norm_economy = max(0, min(1, (12 - row['bowling_economy']) / 6))
                rating = norm_economy
            
        elif role == 'All-rounder':
            # For all-rounders, combine batting and bowling
            batting_rating = 0
            bowling_rating = 0
            
            # Batting component
            norm_avg = min(row['batting_avg'] / 40, 1) if row['batting_avg'] > 0 else 0
            norm_sr = min(row['strike_rate'] / 180, 1) if row['strike_rate'] > 0 else 0
            batting_rating = 0.5 * norm_avg + 0.5 * norm_sr
            
            # Bowling component
            if row['bowling_economy'] > 0:
                # Invert the scale: lower economy is better
                norm_economy = max(0, min(1, (12 - row['bowling_economy']) / 6))
                bowling_rating = norm_economy
            
            # Combined rating for all-rounders
            rating = 0.5 * batting_rating + 0.5 * bowling_rating
        
        # Scale to 0-100
        return rating * 100
    
    # Apply rating calculation
    players_df['player_rating'] = players_df.apply(calculate_player_rating, axis=1)
    
    # Save processed players data
    players_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_players.csv"), index=False)
    logger.info(f"Saved processed players data: {len(players_df)} players")
    
    return players_df

def process_matches_data(data):
    """
    Process matches data
    """
    logger.info("Processing matches data")
    
    if 'matches' not in data:
        logger.warning("Matches data not available")
        return None
    
    matches_df = data['matches'].copy()
    
    # Convert date to datetime
    if 'date' in matches_df.columns:
        matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Add run rate columns
    if 'team1_score' in matches_df.columns:
        # Assuming 20 overs for simplicity
        matches_df['team1_run_rate'] = matches_df['team1_score'] / 20
    
    if 'team2_score' in matches_df.columns:
        # Assuming 20 overs for simplicity
        matches_df['team2_run_rate'] = matches_df['team2_score'] / 20
    
    # Calculate match_margin_percentage
    if 'team1_score' in matches_df.columns and 'team2_score' in matches_df.columns:
        def calculate_margin_percentage(row):
            if row['team1_score'] == 0 or row['team2_score'] == 0:
                return 0
            
            higher_score = max(row['team1_score'], row['team2_score'])
            lower_score = min(row['team1_score'], row['team2_score'])
            
            return (higher_score - lower_score) / lower_score * 100
        
        matches_df['margin_percentage'] = matches_df.apply(calculate_margin_percentage, axis=1)
    
    # Add toss factor (did the toss winner also win the match)
    if 'toss_winner' in matches_df.columns and 'winner' in matches_df.columns:
        matches_df['toss_is_match_winner'] = matches_df['toss_winner'] == matches_df['winner']
    
    # Save processed matches data
    matches_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_matches.csv"), index=False)
    logger.info(f"Saved processed matches data: {len(matches_df)} matches")
    
    return matches_df

def process_current_fixtures(data):
    """
    Process current fixtures data
    """
    logger.info("Processing current fixtures data")
    
    if 'current_fixtures' not in data:
        logger.warning("Current fixtures data not available")
        return None
    
    fixtures_df = data['current_fixtures'].copy()
    
    # Convert date to datetime
    if 'date' in fixtures_df.columns:
        fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
    
    # Calculate team win rates based on completed fixtures
    if 'completed' in fixtures_df.columns and 'winner' in fixtures_df.columns:
        completed_fixtures = fixtures_df[fixtures_df['completed'] == True]
        
        # Calculate win counts for each team
        team_stats = {}
        for team in set(fixtures_df['team1'].unique()) | set(fixtures_df['team2'].unique()):
            team_matches = completed_fixtures[(completed_fixtures['team1'] == team) | (completed_fixtures['team2'] == team)]
            team_wins = len(completed_fixtures[completed_fixtures['winner'] == team])
            
            if len(team_matches) > 0:
                win_rate = (team_wins / len(team_matches)) * 100
            else:
                win_rate = 0
                
            team_stats[team] = {
                'matches_played': len(team_matches),
                'wins': team_wins,
                'win_rate': win_rate
            }
        
        # Add a column with current win rate for each team
        fixtures_df['team1_current_win_rate'] = fixtures_df['team1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0))
        fixtures_df['team2_current_win_rate'] = fixtures_df['team2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0))
    
    # For upcoming fixtures, add prediction placeholder
    if 'completed' in fixtures_df.columns:
        fixtures_df['prediction'] = None
    
    # Save processed fixtures data
    fixtures_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_fixtures.csv"), index=False)
    logger.info(f"Saved processed fixtures data: {len(fixtures_df)} fixtures")
    
    return fixtures_df

def process_team_performance(data):
    """
    Process team performance data
    """
    logger.info("Processing team performance data")
    
    if 'team_performance' not in data:
        logger.warning("Team performance data not available")
        return None
    
    performance_df = data['team_performance'].copy()
    
    # Convert string representation of list back to actual list
    if 'current_form' in performance_df.columns:
        import ast
        performance_df['current_form'] = performance_df['current_form'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Calculate additional metrics
    performance_df['overall_rating'] = 0
    
    # Weight different aspects of team performance
    weights = {
        'batting_avg': 0.15,
        'bowling_economy': -0.15,  # negative because lower is better
        'powerplay_runs': 0.1,
        'death_over_economy': -0.1,  # negative because lower is better
        'form_rating': 0.2,
        'win_batting_first': 0.05,
        'win_chasing': 0.05,
        'home_wins': 0.1,
        'away_wins': 0.1
    }
    
    # Normalize each metric to 0-100 scale before applying weights
    for metric, weight in weights.items():
        if metric in performance_df.columns:
            if metric == 'bowling_economy' or metric == 'death_over_economy':
                # For economy, lower is better, so invert the normalization
                min_val = performance_df[metric].min()
                max_val = performance_df[metric].max()
                if max_val > min_val:
                    performance_df[f'{metric}_norm'] = (max_val - performance_df[metric]) / (max_val - min_val) * 100
                else:
                    performance_df[f'{metric}_norm'] = 50  # Default if all values are the same
            else:
                # For other metrics, higher is better
                min_val = performance_df[metric].min()
                max_val = performance_df[metric].max()
                if max_val > min_val:
                    performance_df[f'{metric}_norm'] = (performance_df[metric] - min_val) / (max_val - min_val) * 100
                else:
                    performance_df[f'{metric}_norm'] = 50  # Default if all values are the same
    
    # Calculate overall rating
    for metric, weight in weights.items():
        if f'{metric}_norm' in performance_df.columns:
            performance_df['overall_rating'] += performance_df[f'{metric}_norm'] * abs(weight)
    
    # Normalize overall rating to 0-100
    min_rating = performance_df['overall_rating'].min()
    max_rating = performance_df['overall_rating'].max()
    if max_rating > min_rating:
        performance_df['overall_rating'] = (performance_df['overall_rating'] - min_rating) / (max_rating - min_rating) * 100
    
    # Calculate strength tiers
    performance_df['tier'] = pd.qcut(performance_df['overall_rating'], 
                                     q=[0, 0.25, 0.5, 0.75, 1.0], 
                                     labels=['D', 'C', 'B', 'A'])
    
    # Save processed team performance data
    performance_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_team_performance.csv"), index=False)
    logger.info(f"Saved processed team performance data: {len(performance_df)} teams")
    
    return performance_df

def prepare_model_data(data):
    """
    Prepare data for modeling
    """
    logger.info("Preparing data for modeling")
    
    processed_teams = None
    if 'processed_teams' in data:
        processed_teams = data['processed_teams']
    
    processed_players = None
    if 'processed_players' in data:
        processed_players = data['processed_players']
    
    processed_matches = None
    if 'processed_matches' in data:
        processed_matches = data['processed_matches']
    
    processed_team_performance = None
    if 'processed_team_performance' in data:
        processed_team_performance = data['processed_team_performance']
    
    processed_fixtures = None
    if 'processed_fixtures' in data:
        processed_fixtures = data['processed_fixtures']
    
    # If we don't have the processed data, return
    if processed_teams is None or processed_matches is None:
        logger.warning("Cannot prepare model data: missing processed teams or matches data")
        return None
    
    # Create a dataset for team vs team prediction
    # For each possible matchup, create a row with features
    
    teams = processed_teams['name'].unique()
    model_data = []
    
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i != j:  # Teams don't play against themselves
                team1_data = processed_teams[processed_teams['name'] == team1].iloc[0]
                team2_data = processed_teams[processed_teams['name'] == team2].iloc[0]
                
                # Extract features for team1 and team2
                features = {
                    'team1': team1,
                    'team2': team2,
                    'team1_titles': team1_data['titles'],
                    'team2_titles': team2_data['titles'],
                    'team1_win_percentage': team1_data.get('win_percentage', 50),  # Default if not available
                    'team2_win_percentage': team2_data.get('win_percentage', 50),  # Default if not available
                    'team1_recent_form': team1_data.get('recent_form', 50),  # Default if not available
                    'team2_recent_form': team2_data.get('recent_form', 50),  # Default if not available
                    'team1_strength': team1_data.get('team_strength', 50),  # Default if not available
                    'team2_strength': team2_data.get('team_strength', 50)   # Default if not available
                }
                
                # Add current season performance metrics if available
                if processed_team_performance is not None:
                    team1_perf = processed_team_performance[processed_team_performance['name'] == team1]
                    team2_perf = processed_team_performance[processed_team_performance['name'] == team2]
                    
                    if len(team1_perf) > 0:
                        features['team1_batting_avg'] = team1_perf.iloc[0].get('batting_avg', 0)
                        features['team1_bowling_economy'] = team1_perf.iloc[0].get('bowling_economy', 0)
                        features['team1_form_rating'] = team1_perf.iloc[0].get('form_rating', 0)
                        features['team1_overall_rating'] = team1_perf.iloc[0].get('overall_rating', 0)
                    
                    if len(team2_perf) > 0:
                        features['team2_batting_avg'] = team2_perf.iloc[0].get('batting_avg', 0)
                        features['team2_bowling_economy'] = team2_perf.iloc[0].get('bowling_economy', 0)
                        features['team2_form_rating'] = team2_perf.iloc[0].get('form_rating', 0)
                        features['team2_overall_rating'] = team2_perf.iloc[0].get('overall_rating', 0)
                
                # Calculate head-to-head stats if we have match data
                if processed_matches is not None:
                    # Get all matches between these two teams
                    h2h_matches = processed_matches[
                        ((processed_matches['team1'] == team1) & (processed_matches['team2'] == team2)) |
                        ((processed_matches['team1'] == team2) & (processed_matches['team2'] == team1))
                    ]
                    
                    # Calculate head-to-head win percentage for team1
                    team1_h2h_wins = len(h2h_matches[h2h_matches['winner'] == team1])
                    if len(h2h_matches) > 0:
                        team1_h2h_win_pct = team1_h2h_wins / len(h2h_matches) * 100
                    else:
                        team1_h2h_win_pct = 50  # Default to 50% if no matches
                    
                    features['team1_h2h_win_percentage'] = team1_h2h_win_pct
                    features['team2_h2h_win_percentage'] = 100 - team1_h2h_win_pct
                
                # Add current season head-to-head if available
                if processed_fixtures is not None:
                    current_h2h = processed_fixtures[
                        (processed_fixtures['completed'] == True) &
                        (
                            ((processed_fixtures['team1'] == team1) & (processed_fixtures['team2'] == team2)) |
                            ((processed_fixtures['team1'] == team2) & (processed_fixtures['team2'] == team1))
                        )
                    ]
                    
                    if len(current_h2h) > 0:
                        team1_current_h2h_wins = len(current_h2h[current_h2h['winner'] == team1])
                        features['team1_current_h2h_wins'] = team1_current_h2h_wins
                        features['team2_current_h2h_wins'] = len(current_h2h) - team1_current_h2h_wins
                    else:
                        features['team1_current_h2h_wins'] = 0
                        features['team2_current_h2h_wins'] = 0
                
                # Add player strength if we have player data
                if processed_players is not None:
                    # Get players for each team
                    team1_players = processed_players[processed_players['team'] == team1]
                    team2_players = processed_players[processed_players['team'] == team2]
                    
                    # Calculate average player rating for each team
                    if len(team1_players) > 0:
                        features['team1_avg_player_rating'] = team1_players['player_rating'].mean()
                    else:
                        features['team1_avg_player_rating'] = 50  # Default if no players
                    
                    if len(team2_players) > 0:
                        features['team2_avg_player_rating'] = team2_players['player_rating'].mean()
                    else:
                        features['team2_avg_player_rating'] = 50  # Default if no players
                    
                    # Calculate average rating by role
                    for role in ['Batsman', 'Bowler', 'All-rounder']:
                        # Team 1
                        team1_role_players = team1_players[team1_players['role'] == role]
                        if len(team1_role_players) > 0:
                            features[f'team1_avg_{role.lower()}_rating'] = team1_role_players['player_rating'].mean()
                        else:
                            features[f'team1_avg_{role.lower()}_rating'] = 50  # Default if no players
                        
                        # Team 2
                        team2_role_players = team2_players[team2_players['role'] == role]
                        if len(team2_role_players) > 0:
                            features[f'team2_avg_{role.lower()}_rating'] = team2_role_players['player_rating'].mean()
                        else:
                            features[f'team2_avg_{role.lower()}_rating'] = 50  # Default if no players
                
                # Calculate win probability based on the features
                # A simple way is to use team strength and head-to-head records
                team1_win_prob = 0.25 * features.get('team1_strength', 50) / 100
                team1_win_prob += 0.2 * features.get('team1_h2h_win_percentage', 50) / 100
                team1_win_prob += 0.15 * features.get('team1_recent_form', 50) / 100
                team1_win_prob += 0.1 * features.get('team1_avg_player_rating', 50) / 100
                
                # Add current season performance to win probability if available
                if 'team1_overall_rating' in features:
                    team1_win_prob += 0.3 * features.get('team1_overall_rating', 50) / 100
                
                team1_win_prob = min(max(team1_win_prob, 0.1), 0.9)  # Clip between 0.1 and 0.9
                features['team1_win_probability'] = team1_win_prob * 100
                features['team2_win_probability'] = (1 - team1_win_prob) * 100
                
                model_data.append(features)
    
    model_df = pd.DataFrame(model_data)
    
    # Save model data
    model_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "model_data.csv"), index=False)
    logger.info(f"Saved model data: {len(model_df)} team matchups")
    
    return model_df

def main():
    """
    Main function for data processing
    """
    logger.info("Starting data processing")
    
    # Load raw data
    data = load_data()
    
    # Process each type of data
    processed_data = {}
    
    # Process teams data
    processed_data['processed_teams'] = process_teams_data(data)
    
    # Process players data
    processed_data['processed_players'] = process_players_data(data)
    
    # Process matches data
    processed_data['processed_matches'] = process_matches_data(data)
    
    # Process current fixtures data
    processed_data['processed_fixtures'] = process_current_fixtures(data)
    
    # Process team performance data
    processed_data['processed_team_performance'] = process_team_performance(data)
    
    # Prepare data for modeling
    model_data = prepare_model_data(processed_data)
    
    logger.info("Data processing completed")

if __name__ == "__main__":
    main() 