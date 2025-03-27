import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

def load_player_data():
    """
    Load player data from raw data directory
    """
    logger.info("Loading player data")
    
    players_file = os.path.join(RAW_DATA_DIR, "ipl_players.csv")
    if not os.path.exists(players_file):
        logger.error(f"Players data file not found: {players_file}")
        return None
    
    players = pd.read_csv(players_file)
    logger.info(f"Loaded players data: {len(players)} players")
    
    return players

def load_matches_data():
    """
    Load matches data from raw data directory
    """
    logger.info("Loading matches data")
    
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
                matches = pd.concat(matches_list, ignore_index=True)
                logger.info(f"Loaded matches data: {len(matches)} matches")
                return matches
            else:
                logger.warning("No match files found in the matches directory")
                return None
        except Exception as e:
            logger.error(f"Error loading matches data: {str(e)}")
            return None
    else:
        logger.warning(f"Matches directory not found: {matches_dir}")
        return None

def calculate_player_impact_scores(players_df, matches_df):
    """
    Calculate impact scores for players based on their performance in matches
    
    This function analyzes player performance data to determine how much impact
    each player has on match outcomes. It considers factors like:
    - Player of the match awards
    - Batting performance (runs scored, strike rate)
    - Bowling performance (wickets taken, economy rate)
    - Performance in winning matches vs losing matches
    - Performance in high-pressure situations (playoffs, finals)
    """
    logger.info("Calculating player impact scores")
    
    if players_df is None or matches_df is None:
        logger.error("Player or match data not available")
        return None
    
    # Create a copy of the players dataframe to add impact scores
    players_impact = players_df.copy()
    
    # Initialize impact score columns
    players_impact['overall_impact'] = 0.0
    players_impact['batting_impact'] = 0.0
    players_impact['bowling_impact'] = 0.0
    players_impact['fielding_impact'] = 0.0
    players_impact['leadership_impact'] = 0.0
    players_impact['clutch_performance'] = 0.0  # Performance in high-pressure situations
    
    # Calculate impact scores based on available data
    # For this example, we'll use a simplified approach
    
    # 1. Player of the match awards
    if 'player_of_match' in matches_df.columns:
        potm_counts = matches_df['player_of_match'].value_counts()
        for player_name, count in potm_counts.items():
            # Find the player in our players dataframe
            player_idx = players_impact[players_impact['name'] == player_name].index
            if len(player_idx) > 0:
                # Add impact points for player of the match awards
                players_impact.loc[player_idx, 'overall_impact'] += count * 5.0
                players_impact.loc[player_idx, 'clutch_performance'] += count * 2.0
    
    # 2. Batting impact (if batting stats are available)
    if 'batting_avg' in players_impact.columns and 'batting_sr' in players_impact.columns:
        # Normalize batting average and strike rate
        max_avg = players_impact['batting_avg'].max()
        max_sr = players_impact['batting_sr'].max()
        
        if max_avg > 0 and max_sr > 0:
            players_impact['batting_impact'] = (
                (players_impact['batting_avg'] / max_avg) * 0.6 +
                (players_impact['batting_sr'] / max_sr) * 0.4
            ) * 10.0
    
    # 3. Bowling impact (if bowling stats are available)
    if 'bowling_avg' in players_impact.columns and 'bowling_economy' in players_impact.columns:
        # For bowling, lower values are better, so invert the normalization
        min_avg = players_impact['bowling_avg'].min()
        min_economy = players_impact['bowling_economy'].min()
        
        if min_avg > 0 and min_economy > 0:
            players_impact['bowling_impact'] = (
                (min_avg / players_impact['bowling_avg']) * 0.6 +
                (min_economy / players_impact['bowling_economy']) * 0.4
            ) * 10.0
    
    # 4. Leadership impact (captains and experienced players)
    if 'is_captain' in players_impact.columns:
        players_impact.loc[players_impact['is_captain'] == True, 'leadership_impact'] += 8.0
    
    if 'experience_years' in players_impact.columns:
        max_exp = players_impact['experience_years'].max()
        if max_exp > 0:
            players_impact['leadership_impact'] += (players_impact['experience_years'] / max_exp) * 5.0
    
    # Calculate overall impact as a weighted sum of all impact factors
    players_impact['overall_impact'] = (
        players_impact['batting_impact'] * 0.3 +
        players_impact['bowling_impact'] * 0.3 +
        players_impact['fielding_impact'] * 0.1 +
        players_impact['leadership_impact'] * 0.2 +
        players_impact['clutch_performance'] * 0.1
    )
    
    # Normalize overall impact to a 0-100 scale
    max_impact = players_impact['overall_impact'].max()
    if max_impact > 0:
        players_impact['overall_impact'] = (players_impact['overall_impact'] / max_impact) * 100.0
    
    logger.info("Player impact scores calculated successfully")
    
    return players_impact

def identify_key_players(players_impact, team_name=None, top_n=5):
    """
    Identify key players overall or for a specific team
    """
    logger.info(f"Identifying key players{' for ' + team_name if team_name else ''}")
    
    if players_impact is None:
        logger.error("Player impact data not available")
        return None
    
    # Filter by team if specified
    if team_name and 'team' in players_impact.columns:
        team_players = players_impact[players_impact['team'] == team_name]
        if len(team_players) == 0:
            logger.warning(f"No players found for team: {team_name}")
            return None
        
        # Get top players by overall impact
        key_players = team_players.sort_values('overall_impact', ascending=False).head(top_n)
    else:
        # Get top players overall
        key_players = players_impact.sort_values('overall_impact', ascending=False).head(top_n)
    
    logger.info(f"Identified {len(key_players)} key players")
    
    return key_players

def calculate_team_player_strength(players_impact):
    """
    Calculate team strength based on player impact scores
    """
    logger.info("Calculating team strength based on player impact")
    
    if players_impact is None or 'team' not in players_impact.columns:
        logger.error("Player impact data not available or missing team information")
        return None
    
    # Group by team and calculate average impact scores
    team_strength = players_impact.groupby('team').agg({
        'overall_impact': 'mean',
        'batting_impact': 'mean',
        'bowling_impact': 'mean',
        'leadership_impact': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    team_strength = team_strength.rename(columns={
        'overall_impact': 'player_strength',
        'batting_impact': 'batting_strength',
        'bowling_impact': 'bowling_strength',
        'leadership_impact': 'leadership_strength'
    })
    
    logger.info(f"Calculated player-based strength for {len(team_strength)} teams")
    
    return team_strength

def save_player_impact_data(players_impact, team_strength):
    """
    Save player impact data to processed data directory
    """
    logger.info("Saving player impact data")
    
    if players_impact is not None:
        # Save player impact data
        player_impact_file = os.path.join(PROCESSED_DATA_DIR, "player_impact.csv")
        players_impact.to_csv(player_impact_file, index=False)
        logger.info(f"Saved player impact data to: {player_impact_file}")
    
    if team_strength is not None:
        # Save team strength data
        team_strength_file = os.path.join(PROCESSED_DATA_DIR, "team_player_strength.csv")
        team_strength.to_csv(team_strength_file, index=False)
        logger.info(f"Saved team player strength data to: {team_strength_file}")

def analyze_player_impact():
    """
    Main function to analyze player impact on match outcomes
    """
    logger.info("Starting player impact analysis")
    
    # Load player and match data
    players_df = load_player_data()
    matches_df = load_matches_data()
    
    # Calculate player impact scores
    players_impact = calculate_player_impact_scores(players_df, matches_df)
    
    # Calculate team strength based on player impact
    team_strength = calculate_team_player_strength(players_impact)
    
    # Save processed data
    save_player_impact_data(players_impact, team_strength)
    
    logger.info("Player impact analysis completed")
    
    return players_impact, team_strength

if __name__ == "__main__":
    analyze_player_impact()