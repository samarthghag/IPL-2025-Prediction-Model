import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
RESULTS_DIR = os.path.join(BASE_DIR, 'models', 'results')

def load_matches_data():
    """
    Load matches data from raw data directory
    """
    logger.info("Loading matches data for time series analysis")
    
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

def load_processed_matches():
    """
    Load processed matches data if available
    """
    logger.info("Loading processed matches data")
    
    processed_matches_file = os.path.join(PROCESSED_DATA_DIR, "processed_matches.csv")
    if os.path.exists(processed_matches_file):
        try:
            matches = pd.read_csv(processed_matches_file)
            logger.info(f"Loaded processed matches data: {len(matches)} matches")
            return matches
        except Exception as e:
            logger.error(f"Error loading processed matches data: {str(e)}")
            return None
    else:
        logger.warning(f"Processed matches file not found: {processed_matches_file}")
        return None

def calculate_team_performance_over_time(matches_df):
    """
    Calculate team performance metrics over time (by season)
    """
    logger.info("Calculating team performance over time")
    
    if matches_df is None:
        logger.error("Matches data not available for time series analysis")
        return None
    
    # Ensure date column is in datetime format
    if 'date' in matches_df.columns:
        try:
            matches_df['date'] = pd.to_datetime(matches_df['date'])
            matches_df['season'] = matches_df['date'].dt.year
        except Exception as e:
            logger.warning(f"Error converting date column: {str(e)}")
            if 'season' not in matches_df.columns:
                logger.error("No season information available for time series analysis")
                return None
    elif 'season' not in matches_df.columns:
        logger.error("No date or season information available for time series analysis")
        return None
    
    # Get list of all teams
    all_teams = set()
    if 'team1' in matches_df.columns and 'team2' in matches_df.columns:
        all_teams.update(matches_df['team1'].unique())
        all_teams.update(matches_df['team2'].unique())
    else:
        logger.error("Team information not available in matches data")
        return None
    
    # Initialize dataframe to store team performance by season
    seasons = sorted(matches_df['season'].unique())
    team_performance = []
    
    # Calculate performance metrics for each team in each season
    for team in all_teams:
        for season in seasons:
            # Filter matches for this team and season
            team_season_matches = matches_df[
                ((matches_df['team1'] == team) | (matches_df['team2'] == team)) &
                (matches_df['season'] == season)
            ]
            
            if len(team_season_matches) == 0:
                continue
            
            # Calculate number of matches played
            matches_played = len(team_season_matches)
            
            # Calculate number of wins
            if 'winner' in team_season_matches.columns:
                wins = len(team_season_matches[team_season_matches['winner'] == team])
            else:
                wins = 0
            
            # Calculate win percentage
            win_percentage = (wins / matches_played) * 100 if matches_played > 0 else 0
            
            # Calculate average runs scored and conceded
            runs_scored = []
            runs_conceded = []
            
            for _, match in team_season_matches.iterrows():
                if match['team1'] == team and 'team1_score' in match:
                    runs_scored.append(match['team1_score'])
                    if 'team2_score' in match:
                        runs_conceded.append(match['team2_score'])
                elif match['team2'] == team and 'team2_score' in match:
                    runs_scored.append(match['team2_score'])
                    if 'team1_score' in match:
                        runs_conceded.append(match['team1_score'])
            
            avg_runs_scored = np.mean(runs_scored) if runs_scored else 0
            avg_runs_conceded = np.mean(runs_conceded) if runs_conceded else 0
            
            # Calculate net run rate (simplified)
            net_run_rate = avg_runs_scored - avg_runs_conceded
            
            # Add to performance data
            team_performance.append({
                'team': team,
                'season': season,
                'matches_played': matches_played,
                'wins': wins,
                'win_percentage': win_percentage,
                'avg_runs_scored': avg_runs_scored,
                'avg_runs_conceded': avg_runs_conceded,
                'net_run_rate': net_run_rate
            })
    
    # Convert to DataFrame
    team_performance_df = pd.DataFrame(team_performance)
    
    logger.info(f"Calculated performance over time for {len(all_teams)} teams across {len(seasons)} seasons")
    
    return team_performance_df

def calculate_team_trends(team_performance_df):
    """
    Calculate trends in team performance over time
    """
    logger.info("Calculating team performance trends")
    
    if team_performance_df is None or len(team_performance_df) == 0:
        logger.error("Team performance data not available for trend analysis")
        return None
    
    # Get list of teams and seasons
    teams = team_performance_df['team'].unique()
    seasons = sorted(team_performance_df['season'].unique())
    
    # Initialize dataframe to store team trends
    team_trends = []
    
    # Calculate trends for each team
    for team in teams:
        # Get team performance data sorted by season
        team_data = team_performance_df[team_performance_df['team'] == team].sort_values('season')
        
        if len(team_data) < 2:
            # Need at least 2 seasons to calculate trend
            continue
        
        # Calculate trend in win percentage
        win_pct_values = team_data['win_percentage'].values
        win_pct_trend = np.polyfit(range(len(win_pct_values)), win_pct_values, 1)[0]
        
        # Calculate trend in net run rate
        if 'net_run_rate' in team_data.columns:
            nrr_values = team_data['net_run_rate'].values
            nrr_trend = np.polyfit(range(len(nrr_values)), nrr_values, 1)[0]
        else:
            nrr_trend = 0
        
        # Calculate recent form (last 2 seasons)
        recent_data = team_data.tail(2)
        recent_win_pct = recent_data['win_percentage'].mean() if len(recent_data) > 0 else 0
        
        # Calculate overall performance
        overall_win_pct = team_data['win_percentage'].mean()
        
        # Determine trend direction
        if win_pct_trend > 1:
            trend_direction = 'Improving'
        elif win_pct_trend < -1:
            trend_direction = 'Declining'
        else:
            trend_direction = 'Stable'
        
        # Add to trends data
        team_trends.append({
            'team': team,
            'win_pct_trend': win_pct_trend,
            'nrr_trend': nrr_trend,
            'recent_win_pct': recent_win_pct,
            'overall_win_pct': overall_win_pct,
            'trend_direction': trend_direction,
            'seasons_analyzed': len(team_data),
            'last_season': team_data['season'].max()
        })
    
    # Convert to DataFrame
    team_trends_df = pd.DataFrame(team_trends)
    
    logger.info(f"Calculated performance trends for {len(team_trends_df)} teams")
    
    return team_trends_df

def plot_team_performance_trends(team_performance_df, team_trends_df):
    """
    Plot team performance trends over time
    """
    logger.info("Plotting team performance trends")
    
    if team_performance_df is None or len(team_performance_df) == 0:
        logger.error("Team performance data not available for plotting")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plot win percentage trends for all teams
    plt.figure(figsize=(14, 8))
    
    for team in team_performance_df['team'].unique():
        team_data = team_performance_df[team_performance_df['team'] == team].sort_values('season')
        if len(team_data) > 1:  # Need at least 2 points to plot a line
            plt.plot(team_data['season'], team_data['win_percentage'], marker='o', label=team)
    
    plt.title('Team Win Percentage Trends Over Seasons', fontsize=16)
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Win Percentage', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, 'team_win_percentage_trends.png'), dpi=300)
    plt.close()
    
    # Plot net run rate trends for all teams
    if 'net_run_rate' in team_performance_df.columns:
        plt.figure(figsize=(14, 8))
        
        for team in team_performance_df['team'].unique():
            team_data = team_performance_df[team_performance_df['team'] == team].sort_values('season')
            if len(team_data) > 1:  # Need at least 2 points to plot a line
                plt.plot(team_data['season'], team_data['net_run_rate'], marker='o', label=team)
        
        plt.title('Team Net Run Rate Trends Over Seasons', fontsize=16)
        plt.xlabel('Season', fontsize=14)
        plt.ylabel('Net Run Rate', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(RESULTS_DIR, 'team_net_run_rate_trends.png'), dpi=300)
        plt.close()
    
    # Plot trend direction distribution
    if team_trends_df is not None and 'trend_direction' in team_trends_df.columns:
        plt.figure(figsize=(10, 6))
        
        trend_counts = team_trends_df['trend_direction'].value_counts()
        sns.barplot(x=trend_counts.index, y=trend_counts.values)
        
        plt.title('Distribution of Team Performance Trends', fontsize=16)
        plt.xlabel('Trend Direction', fontsize=14)
        plt.ylabel('Number of Teams', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(RESULTS_DIR, 'team_trend_distribution.png'), dpi=300)
        plt.close()
    
    logger.info("Team performance trend plots saved to results directory")

def save_time_series_data(team_performance_df, team_trends_df):
    """
    Save time series analysis data
    """
    logger.info("Saving time series analysis data")
    
    if team_performance_df is not None:
        # Save team performance over time data
        performance_file = os.path.join(PROCESSED_DATA_DIR, "team_performance_time_series.csv")
        team_performance_df.to_csv(performance_file, index=False)
        logger.info(f"Saved team performance time series data to: {performance_file}")
    
    if team_trends_df is not None:
        # Save team trends data
        trends_file = os.path.join(PROCESSED_DATA_DIR, "team_performance_trends.csv")
        team_trends_df.to_csv(trends_file, index=False)
        logger.info(f"Saved team performance trends data to: {trends_file}")
        
        # Also save to results directory for visualization
        results_file = os.path.join(RESULTS_DIR, "team_performance_trends.csv")
        team_trends_df.to_csv(results_file, index=False)

def analyze_time_series():
    """
    Main function to analyze team performance over time
    """
    logger.info("Starting time series analysis of team performance")
    
    # Load matches data
    matches_df = load_processed_matches()
    if matches_df is None:
        matches_df = load_matches_data()
    
    if matches_df is None:
        logger.error("No matches data available for time series analysis")
        return None, None
    
    # Calculate team performance over time
    team_performance_df = calculate_team_performance_over_time(matches_df)
    
    # Calculate team performance trends
    team_trends_df = calculate_team_trends(team_performance_df)
    
    # Plot team performance trends
    plot_team_performance_trends(team_performance_df, team_trends_df)
    
    # Save time series data
    save_time_series_data(team_performance_df, team_trends_df)
    
    logger.info("Time series analysis completed")
    
    return team_performance_df, team_trends_df

if __name__ == "__main__":
    analyze_time_series()