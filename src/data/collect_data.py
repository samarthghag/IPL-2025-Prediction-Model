import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

logger.info(f"Data directory path: {DATA_DIR}")
logger.info(f"Raw data directory path: {RAW_DATA_DIR}")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_match_data():
    """
    Download IPL match data from Cricsheet
    """
    logger.info("Downloading IPL match data")
    
    # URL for IPL data from Cricsheet (in CSV format)
    url = "https://cricsheet.org/downloads/ipl_csv.zip"
    
    try:
        # Download the zip file
        response = requests.get(url)
        if response.status_code == 200:
            zip_file = os.path.join(RAW_DATA_DIR, "ipl_csv.zip")
            with open(zip_file, 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(RAW_DATA_DIR, "ipl_csv"))
            
            logger.info(f"Downloaded and extracted IPL data to {RAW_DATA_DIR}/ipl_csv")
            return True
        else:
            logger.error(f"Failed to download data: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        return False

def download_team_data():
    """
    Download current IPL team and player data
    """
    logger.info("Downloading current IPL team data")
    
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
        "Sunrisers Hyderabad"
    ]
    
    team_data = []
    
    # In a real scenario, we would scrape this data from ESPNCricinfo or similar
    # For this example, we'll create sample data
    for team in teams:
        team_data.append({
            "name": team,
            "titles": get_team_titles(team),
            "captain": get_team_captain(team),
            "home_ground": get_team_home_ground(team)
        })
    
    # Save the team data
    df_teams = pd.DataFrame(team_data)
    df_teams.to_csv(os.path.join(RAW_DATA_DIR, "ipl_teams.csv"), index=False)
    logger.info(f"Saved team data to {RAW_DATA_DIR}/ipl_teams.csv")
    
    return True

def get_team_titles(team):
    """Return number of IPL titles for each team"""
    titles = {
        "Mumbai Indians": 5,
        "Chennai Super Kings": 5,
        "Kolkata Knight Riders": 3,
        "Gujarat Titans": 1,
        "Rajasthan Royals": 1,
        "Sunrisers Hyderabad": 1,
        "Royal Challengers Bangalore": 0,
        "Delhi Capitals": 0,
        "Punjab Kings": 0,
        "Lucknow Super Giants": 0
    }
    return titles.get(team, 0)

def get_team_captain(team):
    """Return current captain for each team"""
    captains = {
        "Mumbai Indians": "Hardik Pandya",
        "Chennai Super Kings": "Ruturaj Gaikwad",
        "Kolkata Knight Riders": "Shreyas Iyer",
        "Gujarat Titans": "Shubman Gill",
        "Rajasthan Royals": "Sanju Samson",
        "Sunrisers Hyderabad": "Pat Cummins",
        "Royal Challengers Bangalore": "Faf du Plessis",
        "Delhi Capitals": "Rishabh Pant",
        "Punjab Kings": "Shikhar Dhawan",
        "Lucknow Super Giants": "KL Rahul"
    }
    return captains.get(team, "Unknown")

def get_team_home_ground(team):
    """Return home ground for each team"""
    grounds = {
        "Mumbai Indians": "Wankhede Stadium",
        "Chennai Super Kings": "M. A. Chidambaram Stadium",
        "Kolkata Knight Riders": "Eden Gardens",
        "Gujarat Titans": "Narendra Modi Stadium",
        "Rajasthan Royals": "Sawai Mansingh Stadium",
        "Sunrisers Hyderabad": "Rajiv Gandhi International Stadium",
        "Royal Challengers Bangalore": "M. Chinnaswamy Stadium",
        "Delhi Capitals": "Arun Jaitley Stadium",
        "Punjab Kings": "PCA Stadium, Mohali",
        "Lucknow Super Giants": "BRSABV Ekana Cricket Stadium"
    }
    return grounds.get(team, "Unknown")

def download_player_data():
    """
    Download player statistics
    """
    logger.info("Downloading player statistics")
    
    # In a real scenario, this would be scraped from a cricket statistics website
    # For this example, we'll create sample data for key players
    
    players = [
        {"name": "Virat Kohli", "team": "Royal Challengers Bangalore", "role": "Batsman", 
         "batting_avg": 37.21, "strike_rate": 130.02, "bowling_economy": 8.91},
        {"name": "Rohit Sharma", "team": "Mumbai Indians", "role": "Batsman", 
         "batting_avg": 30.31, "strike_rate": 130.54, "bowling_economy": 7.82},
        {"name": "Jasprit Bumrah", "team": "Mumbai Indians", "role": "Bowler", 
         "batting_avg": 5.12, "strike_rate": 90.23, "bowling_economy": 6.74},
        {"name": "Ravindra Jadeja", "team": "Chennai Super Kings", "role": "All-rounder", 
         "batting_avg": 26.54, "strike_rate": 140.13, "bowling_economy": 7.62},
        {"name": "KL Rahul", "team": "Lucknow Super Giants", "role": "Batsman", 
         "batting_avg": 47.33, "strike_rate": 135.29, "bowling_economy": 0},
        {"name": "Rishabh Pant", "team": "Delhi Capitals", "role": "Wicket-keeper", 
         "batting_avg": 35.21, "strike_rate": 147.97, "bowling_economy": 0},
        {"name": "Pat Cummins", "team": "Sunrisers Hyderabad", "role": "Bowler", 
         "batting_avg": 15.67, "strike_rate": 141.23, "bowling_economy": 8.41},
        {"name": "Jos Buttler", "team": "Rajasthan Royals", "role": "Wicket-keeper", 
         "batting_avg": 38.06, "strike_rate": 150.56, "bowling_economy": 0},
        {"name": "Andre Russell", "team": "Kolkata Knight Riders", "role": "All-rounder", 
         "batting_avg": 29.36, "strike_rate": 177.88, "bowling_economy": 9.14},
        {"name": "Rashid Khan", "team": "Gujarat Titans", "role": "Bowler", 
         "batting_avg": 16.79, "strike_rate": 153.48, "bowling_economy": 6.38}
    ]
    
    # Add more players - around 10 per team
    more_players = []
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
        "Sunrisers Hyderabad"
    ]
    
    import random
    for team in teams:
        for i in range(9):  # 9 more players per team (plus 1 from the key players)
            role = random.choice(["Batsman", "Bowler", "All-rounder", "Wicket-keeper"])
            
            batting_avg = 0
            strike_rate = 0
            bowling_economy = 0
            
            if role in ["Batsman", "All-rounder", "Wicket-keeper"]:
                batting_avg = round(random.uniform(15, 45), 2)
                strike_rate = round(random.uniform(110, 160), 2)
            
            if role in ["Bowler", "All-rounder"]:
                bowling_economy = round(random.uniform(6, 10), 2)
            
            more_players.append({
                "name": f"Player_{team}_{i}", 
                "team": team, 
                "role": role,
                "batting_avg": batting_avg, 
                "strike_rate": strike_rate, 
                "bowling_economy": bowling_economy
            })
    
    players.extend(more_players)
    
    # Save the player data
    df_players = pd.DataFrame(players)
    df_players.to_csv(os.path.join(RAW_DATA_DIR, "ipl_players.csv"), index=False)
    logger.info(f"Saved player data to {RAW_DATA_DIR}/ipl_players.csv")
    
    return True

def download_venue_data():
    """
    Download venue statistics 
    """
    logger.info("Downloading venue statistics")
    
    venues = [
        {"name": "Wankhede Stadium", "city": "Mumbai", "avg_first_innings": 172, "avg_second_innings": 158},
        {"name": "M. A. Chidambaram Stadium", "city": "Chennai", "avg_first_innings": 168, "avg_second_innings": 152},
        {"name": "Eden Gardens", "city": "Kolkata", "avg_first_innings": 170, "avg_second_innings": 155},
        {"name": "Narendra Modi Stadium", "city": "Ahmedabad", "avg_first_innings": 175, "avg_second_innings": 162},
        {"name": "Sawai Mansingh Stadium", "city": "Jaipur", "avg_first_innings": 167, "avg_second_innings": 156},
        {"name": "Rajiv Gandhi International Stadium", "city": "Hyderabad", "avg_first_innings": 166, "avg_second_innings": 154},
        {"name": "M. Chinnaswamy Stadium", "city": "Bangalore", "avg_first_innings": 180, "avg_second_innings": 165},
        {"name": "Arun Jaitley Stadium", "city": "Delhi", "avg_first_innings": 174, "avg_second_innings": 160},
        {"name": "PCA Stadium", "city": "Mohali", "avg_first_innings": 173, "avg_second_innings": 159},
        {"name": "BRSABV Ekana Cricket Stadium", "city": "Lucknow", "avg_first_innings": 165, "avg_second_innings": 153}
    ]
    
    # Save the venue data
    df_venues = pd.DataFrame(venues)
    df_venues.to_csv(os.path.join(RAW_DATA_DIR, "ipl_venues.csv"), index=False)
    logger.info(f"Saved venue data to {RAW_DATA_DIR}/ipl_venues.csv")
    
    return True

def download_historical_season_data():
    """
    Download historical IPL season data
    """
    logger.info("Downloading historical IPL season data")
    
    seasons = []
    
    # Data for the most recent few seasons
    seasons.append({
        "year": 2024,
        "winner": "Kolkata Knight Riders",
        "runner_up": "Sunrisers Hyderabad",
        "top_run_scorer": "Virat Kohli (Royal Challengers Bangalore)",
        "top_wicket_taker": "Jasprit Bumrah (Mumbai Indians)",
        "most_valuable_player": "Sunil Narine (Kolkata Knight Riders)"
    })
    
    seasons.append({
        "year": 2023,
        "winner": "Chennai Super Kings",
        "runner_up": "Gujarat Titans",
        "top_run_scorer": "Shubman Gill (Gujarat Titans)",
        "top_wicket_taker": "Mohammed Shami (Gujarat Titans)",
        "most_valuable_player": "Shubman Gill (Gujarat Titans)"
    })
    
    seasons.append({
        "year": 2022,
        "winner": "Gujarat Titans",
        "runner_up": "Rajasthan Royals",
        "top_run_scorer": "Jos Buttler (Rajasthan Royals)",
        "top_wicket_taker": "Yuzvendra Chahal (Rajasthan Royals)",
        "most_valuable_player": "Jos Buttler (Rajasthan Royals)"
    })
    
    seasons.append({
        "year": 2021,
        "winner": "Chennai Super Kings",
        "runner_up": "Kolkata Knight Riders",
        "top_run_scorer": "Ruturaj Gaikwad (Chennai Super Kings)",
        "top_wicket_taker": "Harshal Patel (Royal Challengers Bangalore)",
        "most_valuable_player": "Harshal Patel (Royal Challengers Bangalore)"
    })
    
    seasons.append({
        "year": 2020,
        "winner": "Mumbai Indians",
        "runner_up": "Delhi Capitals",
        "top_run_scorer": "KL Rahul (Punjab Kings)",
        "top_wicket_taker": "Kagiso Rabada (Delhi Capitals)",
        "most_valuable_player": "Jofra Archer (Rajasthan Royals)"
    })
    
    # Add more historical data
    for year in range(2019, 2007, -1):
        winner = ""
        runner_up = ""
        
        if year == 2019:
            winner = "Mumbai Indians"
            runner_up = "Chennai Super Kings"
        elif year == 2018:
            winner = "Chennai Super Kings"
            runner_up = "Sunrisers Hyderabad"
        elif year == 2017:
            winner = "Mumbai Indians"
            runner_up = "Rising Pune Supergiant"
        elif year == 2016:
            winner = "Sunrisers Hyderabad"
            runner_up = "Royal Challengers Bangalore"
        elif year == 2015:
            winner = "Mumbai Indians"
            runner_up = "Chennai Super Kings"
        elif year == 2014:
            winner = "Kolkata Knight Riders"
            runner_up = "Kings XI Punjab"
        elif year == 2013:
            winner = "Mumbai Indians"
            runner_up = "Chennai Super Kings"
        elif year == 2012:
            winner = "Kolkata Knight Riders"
            runner_up = "Chennai Super Kings"
        elif year == 2011:
            winner = "Chennai Super Kings"
            runner_up = "Royal Challengers Bangalore"
        elif year == 2010:
            winner = "Chennai Super Kings"
            runner_up = "Mumbai Indians"
        elif year == 2009:
            winner = "Deccan Chargers"
            runner_up = "Royal Challengers Bangalore"
        elif year == 2008:
            winner = "Rajasthan Royals"
            runner_up = "Chennai Super Kings"
            
        seasons.append({
            "year": year,
            "winner": winner,
            "runner_up": runner_up,
            "top_run_scorer": "",  # These would be filled with actual data
            "top_wicket_taker": "",
            "most_valuable_player": ""
        })
    
    # Save the seasons data
    df_seasons = pd.DataFrame(seasons)
    df_seasons.to_csv(os.path.join(RAW_DATA_DIR, "ipl_seasons.csv"), index=False)
    logger.info(f"Saved seasons data to {RAW_DATA_DIR}/ipl_seasons.csv")
    
    return True

def download_current_fixtures():
    """
    Download current IPL fixtures data
    This would typically come from an API, but we'll create sample data
    """
    logger.info("Downloading current IPL fixtures data")
    
    current_year = datetime.now().year
    
    # Create sample fixtures data for the current season
    fixtures = []
    
    # Team names
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
        "Sunrisers Hyderabad"
    ]
    
    # Venues
    venues = [
        "Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens", 
        "Narendra Modi Stadium", "Sawai Mansingh Stadium", "Rajiv Gandhi International Stadium",
        "M. Chinnaswamy Stadium", "Arun Jaitley Stadium", "PCA Stadium, Mohali", 
        "BRSABV Ekana Cricket Stadium"
    ]
    
    # Sample 2024 IPL matches (these would be actual fixtures in a real system)
    # Format: date, team1, team2, venue, completed (True if match has been played)
    fixture_data = [
        (f"{current_year}-03-22", "Chennai Super Kings", "Royal Challengers Bangalore", "M. A. Chidambaram Stadium", True),
        (f"{current_year}-03-23", "Punjab Kings", "Delhi Capitals", "PCA Stadium, Mohali", True),
        (f"{current_year}-03-23", "Kolkata Knight Riders", "Sunrisers Hyderabad", "Eden Gardens", True),
        (f"{current_year}-03-24", "Rajasthan Royals", "Lucknow Super Giants", "Sawai Mansingh Stadium", True),
        (f"{current_year}-03-24", "Gujarat Titans", "Mumbai Indians", "Narendra Modi Stadium", True),
        (f"{current_year}-03-25", "Royal Challengers Bangalore", "Punjab Kings", "M. Chinnaswamy Stadium", True),
        (f"{current_year}-03-26", "Chennai Super Kings", "Gujarat Titans", "M. A. Chidambaram Stadium", True),
        (f"{current_year}-03-27", "Sunrisers Hyderabad", "Mumbai Indians", "Rajiv Gandhi International Stadium", True),
        (f"{current_year}-03-28", "Rajasthan Royals", "Delhi Capitals", "Sawai Mansingh Stadium", True),
        (f"{current_year}-03-29", "Royal Challengers Bangalore", "Kolkata Knight Riders", "M. Chinnaswamy Stadium", True),
        (f"{current_year}-03-30", "Lucknow Super Giants", "Punjab Kings", "BRSABV Ekana Cricket Stadium", True),
        (f"{current_year}-03-31", "Gujarat Titans", "Sunrisers Hyderabad", "Narendra Modi Stadium", True),
        (f"{current_year}-04-01", "Delhi Capitals", "Chennai Super Kings", "Arun Jaitley Stadium", True),
        (f"{current_year}-04-02", "Mumbai Indians", "Rajasthan Royals", "Wankhede Stadium", True),
        (f"{current_year}-04-03", "Royal Challengers Bangalore", "Lucknow Super Giants", "M. Chinnaswamy Stadium", True),
        (f"{current_year}-04-04", "Delhi Capitals", "Kolkata Knight Riders", "Arun Jaitley Stadium", True),
        (f"{current_year}-04-05", "Sunrisers Hyderabad", "Chennai Super Kings", "Rajiv Gandhi International Stadium", True),
        # Upcoming fixtures (not yet played)
        (f"{current_year}-04-06", "Mumbai Indians", "Delhi Capitals", "Wankhede Stadium", False),
        (f"{current_year}-04-07", "Lucknow Super Giants", "Gujarat Titans", "BRSABV Ekana Cricket Stadium", False),
        (f"{current_year}-04-08", "Rajasthan Royals", "Royal Challengers Bangalore", "Sawai Mansingh Stadium", False),
        (f"{current_year}-04-09", "Kolkata Knight Riders", "Punjab Kings", "Eden Gardens", False),
        (f"{current_year}-04-10", "Chennai Super Kings", "Mumbai Indians", "M. A. Chidambaram Stadium", False),
    ]
    
    # Sample results for completed matches (would be real results in production)
    results = {
        0: {"winner": "Chennai Super Kings", "margin": 35},
        1: {"winner": "Delhi Capitals", "margin": 8},
        2: {"winner": "Kolkata Knight Riders", "margin": 57},
        3: {"winner": "Rajasthan Royals", "margin": 15},
        4: {"winner": "Mumbai Indians", "margin": 6},
        5: {"winner": "Royal Challengers Bangalore", "margin": 26},
        6: {"winner": "Gujarat Titans", "margin": 34},
        7: {"winner": "Sunrisers Hyderabad", "margin": 43},
        8: {"winner": "Rajasthan Royals", "margin": 20},
        9: {"winner": "Kolkata Knight Riders", "margin": 19},
        10: {"winner": "Punjab Kings", "margin": 25},
        11: {"winner": "Sunrisers Hyderabad", "margin": 7},
        12: {"winner": "Chennai Super Kings", "margin": 52},
        13: {"winner": "Rajasthan Royals", "margin": 14},
        14: {"winner": "Lucknow Super Giants", "margin": 28},
        15: {"winner": "Kolkata Knight Riders", "margin": 45},
        16: {"winner": "Chennai Super Kings", "margin": 18},
    }
    
    # Generate fixtures with results
    for i, (date, team1, team2, venue, completed) in enumerate(fixture_data):
        fixture = {
            'date': date,
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'completed': completed
        }
        
        if completed and i in results:
            fixture['winner'] = results[i]['winner']
            fixture['margin'] = results[i]['margin']
        
        fixtures.append(fixture)
    
    # Create DataFrame
    fixtures_df = pd.DataFrame(fixtures)
    
    # Save to CSV
    fixtures_file = os.path.join(RAW_DATA_DIR, "current_fixtures.csv")
    fixtures_df.to_csv(fixtures_file, index=False)
    logger.info(f"Saved current fixtures data: {len(fixtures_df)} matches")
    
    return fixtures_df

def collect_team_performance_data():
    """
    Collect detailed team performance metrics for the current season
    """
    logger.info("Collecting team performance data")
    
    # Team names
    teams = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
        "Sunrisers Hyderabad"
    ]
    
    # Sample team performance data (would be calculated from real data in production)
    team_data = [
        {"name": "Chennai Super Kings", "batting_avg": 168.3, "bowling_economy": 8.2, "powerplay_runs": 54.2, "death_over_economy": 9.8, "avg_first_innings": 175.2, "avg_second_innings": 165.4, "win_batting_first": 2, "win_chasing": 1, "home_wins": 2, "away_wins": 1, "current_form": [1, 1, 0, 1]},
        {"name": "Delhi Capitals", "batting_avg": 162.1, "bowling_economy": 8.6, "powerplay_runs": 51.6, "death_over_economy": 10.2, "avg_first_innings": 159.8, "avg_second_innings": 163.5, "win_batting_first": 0, "win_chasing": 1, "home_wins": 0, "away_wins": 1, "current_form": [1, 0, 0, 0]},
        {"name": "Gujarat Titans", "batting_avg": 160.5, "bowling_economy": 8.4, "powerplay_runs": 49.5, "death_over_economy": 9.9, "avg_first_innings": 158.7, "avg_second_innings": 159.2, "win_batting_first": 1, "win_chasing": 0, "home_wins": 1, "away_wins": 0, "current_form": [0, 1, 0, 0]},
        {"name": "Kolkata Knight Riders", "batting_avg": 178.6, "bowling_economy": 7.8, "powerplay_runs": 58.9, "death_over_economy": 8.7, "avg_first_innings": 182.3, "avg_second_innings": 171.5, "win_batting_first": 2, "win_chasing": 1, "home_wins": 2, "away_wins": 1, "current_form": [1, 0, 1, 1]},
        {"name": "Lucknow Super Giants", "batting_avg": 165.8, "bowling_economy": 8.5, "powerplay_runs": 52.3, "death_over_economy": 9.5, "avg_first_innings": 167.1, "avg_second_innings": 158.3, "win_batting_first": 1, "win_chasing": 0, "home_wins": 0, "away_wins": 1, "current_form": [0, 0, 1, 0]},
        {"name": "Mumbai Indians", "batting_avg": 172.4, "bowling_economy": 8.9, "powerplay_runs": 55.7, "death_over_economy": 10.1, "avg_first_innings": 174.8, "avg_second_innings": 168.2, "win_batting_first": 0, "win_chasing": 1, "home_wins": 0, "away_wins": 1, "current_form": [1, 0, 0, 0]},
        {"name": "Punjab Kings", "batting_avg": 169.6, "bowling_economy": 9.1, "powerplay_runs": 57.8, "death_over_economy": 10.8, "avg_first_innings": 172.3, "avg_second_innings": 161.4, "win_batting_first": 1, "win_chasing": 0, "home_wins": 0, "away_wins": 1, "current_form": [0, 0, 1, 0]},
        {"name": "Rajasthan Royals", "batting_avg": 173.2, "bowling_economy": 8.0, "powerplay_runs": 53.4, "death_over_economy": 9.2, "avg_first_innings": 178.6, "avg_second_innings": 166.7, "win_batting_first": 2, "win_chasing": 1, "home_wins": 2, "away_wins": 1, "current_form": [1, 1, 0, 1]},
        {"name": "Royal Challengers Bangalore", "batting_avg": 176.5, "bowling_economy": 9.2, "powerplay_runs": 56.2, "death_over_economy": 11.3, "avg_first_innings": 180.2, "avg_second_innings": 170.4, "win_batting_first": 1, "win_chasing": 0, "home_wins": 1, "away_wins": 0, "current_form": [0, 1, 0, 0]},
        {"name": "Sunrisers Hyderabad", "batting_avg": 167.8, "bowling_economy": 8.7, "powerplay_runs": 54.8, "death_over_economy": 10.5, "avg_first_innings": 169.3, "avg_second_innings": 162.8, "win_batting_first": 1, "win_chasing": 1, "home_wins": 1, "away_wins": 1, "current_form": [1, 0, 1, 0]}
    ]
    
    # Calculate form rating (last 4 matches)
    for team in team_data:
        team['form_rating'] = sum(team['current_form']) / len(team['current_form']) * 100
        team['current_form'] = str(team['current_form'])  # Convert list to string for CSV storage
    
    # Create DataFrame
    team_performance_df = pd.DataFrame(team_data)
    
    # Save to CSV
    team_file = os.path.join(RAW_DATA_DIR, "team_performance.csv")
    team_performance_df.to_csv(team_file, index=False)
    logger.info(f"Saved team performance data: {len(team_performance_df)} teams")
    
    return team_performance_df

def main():
    """
    Main function to collect all data
    """
    logger.info("Starting data collection")
    
    # Download match data
    download_match_data()
    
    # Download team data
    download_team_data()
    
    # Download player data
    download_player_data()
    
    # Download venue data
    download_venue_data()
    
    # Download historical season data
    download_historical_season_data()
    
    # Download current fixtures
    fixtures_df = download_current_fixtures()
    
    # Collect team performance data
    team_performance_df = collect_team_performance_data()
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    main() 