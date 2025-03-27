import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_venue_data():
    """
    Load venue data from raw data directory
    """
    logger.info("Loading venue data")
    
    venues_file = os.path.join(RAW_DATA_DIR, "ipl_venues.csv")
    if not os.path.exists(venues_file):
        logger.error(f"Venues data file not found: {venues_file}")
        return None
    
    venues = pd.read_csv(venues_file)
    logger.info(f"Loaded venues data: {len(venues)} venues")
    
    return venues

def load_matches_data():
    """
    Load matches data from raw data directory
    """
    logger.info("Loading matches data for weather analysis")
    
    # Try to load processed matches data first
    processed_matches_file = os.path.join(PROCESSED_DATA_DIR, "processed_matches.csv")
    if os.path.exists(processed_matches_file):
        try:
            matches = pd.read_csv(processed_matches_file)
            logger.info(f"Loaded processed matches data: {len(matches)} matches")
            return matches
        except Exception as e:
            logger.error(f"Error loading processed matches data: {str(e)}")
    
    # If processed data not available, try raw data
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

def generate_weather_data(venues_df, matches_df):
    """
    Generate synthetic weather data for matches based on venue location and match date
    
    In a real-world scenario, this would fetch historical weather data from an API
    based on venue location and match date. For this example, we'll generate
    synthetic weather data based on venue location and season.
    """
    logger.info("Generating weather data for matches")
    
    if venues_df is None or matches_df is None:
        logger.error("Venue or match data not available")
        return None
    
    # Ensure date column is in datetime format
    if 'date' in matches_df.columns:
        try:
            matches_df['date'] = pd.to_datetime(matches_df['date'])
            matches_df['month'] = matches_df['date'].dt.month
        except Exception as e:
            logger.warning(f"Error converting date column: {str(e)}")
            matches_df['month'] = 4  # Default to April (IPL season)
    else:
        matches_df['month'] = 4  # Default to April (IPL season)
    
    # Create a copy of matches dataframe to add weather data
    matches_weather = matches_df.copy()
    
    # Initialize weather columns
    matches_weather['temperature'] = np.nan
    matches_weather['humidity'] = np.nan
    matches_weather['wind_speed'] = np.nan
    matches_weather['precipitation'] = np.nan
    matches_weather['weather_condition'] = ''
    
    # Define weather patterns for different regions in India during IPL season (March-May)
    # These are simplified patterns for demonstration purposes
    weather_patterns = {
        'North': {
            3: {'temp_range': (25, 35), 'humidity_range': (30, 50), 'wind_range': (5, 15), 'precip_prob': 0.1},
            4: {'temp_range': (30, 40), 'humidity_range': (25, 45), 'wind_range': (5, 20), 'precip_prob': 0.05},
            5: {'temp_range': (35, 45), 'humidity_range': (20, 40), 'wind_range': (10, 25), 'precip_prob': 0.02}
        },
        'South': {
            3: {'temp_range': (28, 36), 'humidity_range': (60, 80), 'wind_range': (5, 15), 'precip_prob': 0.2},
            4: {'temp_range': (30, 38), 'humidity_range': (65, 85), 'wind_range': (5, 15), 'precip_prob': 0.15},
            5: {'temp_range': (32, 40), 'humidity_range': (70, 90), 'wind_range': (5, 15), 'precip_prob': 0.25}
        },
        'East': {
            3: {'temp_range': (25, 35), 'humidity_range': (50, 70), 'wind_range': (5, 15), 'precip_prob': 0.15},
            4: {'temp_range': (30, 38), 'humidity_range': (55, 75), 'wind_range': (5, 15), 'precip_prob': 0.2},
            5: {'temp_range': (32, 42), 'humidity_range': (60, 80), 'wind_range': (5, 15), 'precip_prob': 0.3}
        },
        'West': {
            3: {'temp_range': (28, 36), 'humidity_range': (40, 60), 'wind_range': (10, 20), 'precip_prob': 0.05},
            4: {'temp_range': (32, 40), 'humidity_range': (35, 55), 'wind_range': (10, 25), 'precip_prob': 0.02},
            5: {'temp_range': (35, 45), 'humidity_range': (30, 50), 'wind_range': (15, 30), 'precip_prob': 0.01}
        },
        'Central': {
            3: {'temp_range': (28, 38), 'humidity_range': (30, 50), 'wind_range': (5, 15), 'precip_prob': 0.1},
            4: {'temp_range': (32, 42), 'humidity_range': (25, 45), 'wind_range': (5, 15), 'precip_prob': 0.05},
            5: {'temp_range': (35, 45), 'humidity_range': (20, 40), 'wind_range': (10, 20), 'precip_prob': 0.02}
        }
    }
    
    # Map venues to regions (simplified)
    venue_regions = {
        'Wankhede Stadium': 'West',
        'M. A. Chidambaram Stadium': 'South',
        'Eden Gardens': 'East',
        'Arun Jaitley Stadium': 'North',
        'M. Chinnaswamy Stadium': 'South',
        'Rajiv Gandhi International Stadium': 'South',
        'Punjab Cricket Association Stadium': 'North',
        'Sawai Mansingh Stadium': 'North',
        'Narendra Modi Stadium': 'West',
        'BRSABV Ekana Cricket Stadium': 'North'
    }
    
    # Weather conditions based on temperature and precipitation
    def get_weather_condition(temp, precip):
        if precip > 0.5:
            return 'Rainy'
        elif precip > 0.1:
            return 'Light Rain'
        elif temp > 40:
            return 'Very Hot'
        elif temp > 35:
            return 'Hot'
        elif temp > 25:
            return 'Warm'
        else:
            return 'Pleasant'
    
    # Generate weather data for each match
    import random
    
    for idx, match in matches_weather.iterrows():
        venue = match.get('venue', '')
        month = match.get('month', 4)  # Default to April if month not available
        
        # Ensure month is within range 3-5 (March-May)
        if month < 3 or month > 5:
            month = 4  # Default to April
        
        # Get region for venue
        region = venue_regions.get(venue, 'Central')  # Default to Central if venue not found
        
        # Get weather pattern for region and month
        pattern = weather_patterns[region][month]
        
        # Generate random weather values based on patterns
        temp = random.uniform(*pattern['temp_range'])
        humidity = random.uniform(*pattern['humidity_range'])
        wind = random.uniform(*pattern['wind_range'])
        precip = random.random() if random.random() < pattern['precip_prob'] else 0
        
        # Determine weather condition
        condition = get_weather_condition(temp, precip)
        
        # Add weather data to match
        matches_weather.loc[idx, 'temperature'] = round(temp, 1)
        matches_weather.loc[idx, 'humidity'] = round(humidity, 1)
        matches_weather.loc[idx, 'wind_speed'] = round(wind, 1)
        matches_weather.loc[idx, 'precipitation'] = round(precip, 2)
        matches_weather.loc[idx, 'weather_condition'] = condition
    
    logger.info("Weather data generated for matches")
    
    return matches_weather

def analyze_weather_impact(matches_weather):
    """
    Analyze the impact of weather conditions on match outcomes
    """
    logger.info("Analyzing weather impact on match outcomes")
    
    if matches_weather is None:
        logger.error("Weather data not available for analysis")
        return None
    
    # Initialize dataframe to store weather impact analysis
    weather_impact = []
    
    # Analyze impact of temperature
    temp_bins = [0, 25, 30, 35, 40, 50]
    temp_labels = ['< 25°C', '25-30°C', '30-35°C', '35-40°C', '> 40°C']
    
    matches_weather['temp_range'] = pd.cut(matches_weather['temperature'], bins=temp_bins, labels=temp_labels)
    
    temp_impact = matches_weather.groupby('temp_range').agg({
        'id': 'count',
        'team1_score': 'mean',
        'team2_score': 'mean'
    }).reset_index()
    
    temp_impact = temp_impact.rename(columns={
        'id': 'matches',
        'team1_score': 'avg_first_innings',
        'team2_score': 'avg_second_innings'
    })
    
    temp_impact['total_runs'] = (temp_impact['avg_first_innings'] + temp_impact['avg_second_innings']) / 2
    temp_impact['factor_type'] = 'Temperature'
    
    weather_impact.append(temp_impact)
    
    # Analyze impact of humidity
    humidity_bins = [0, 30, 50, 70, 90, 100]
    humidity_labels = ['< 30%', '30-50%', '50-70%', '70-90%', '> 90%']
    
    matches_weather['humidity_range'] = pd.cut(matches_weather['humidity'], bins=humidity_bins, labels=humidity_labels)
    
    humidity_impact = matches_weather.groupby('humidity_range').agg({
        'id': 'count',
        'team1_score': 'mean',
        'team2_score': 'mean'
    }).reset_index()
    
    humidity_impact = humidity_impact.rename(columns={
        'id': 'matches',
        'team1_score': 'avg_first_innings',
        'team2_score': 'avg_second_innings'
    })
    
    humidity_impact['total_runs'] = (humidity_impact['avg_first_innings'] + humidity_impact['avg_second_innings']) / 2
    humidity_impact['factor_type'] = 'Humidity'
    
    weather_impact.append(humidity_impact)
    
    # Analyze impact of wind speed
    wind_bins = [0, 5, 10, 15, 20, 30]
    wind_labels = ['< 5 km/h', '5-10 km/h', '10-15 km/h', '15-20 km/h', '> 20 km/h']
    
    matches_weather['wind_range'] = pd.cut(matches_weather['wind_speed'], bins=wind_bins, labels=wind_labels)
    
    wind_impact = matches_weather.groupby('wind_range').agg({
        'id': 'count',
        'team1_score': 'mean',
        'team2_score': 'mean'
    }).reset_index()
    
    wind_impact = wind_impact.rename(columns={
        'id': 'matches',
        'team1_score': 'avg_first_innings',
        'team2_score': 'avg_second_innings'
    })
    
    wind_impact['total_runs'] = (wind_impact['avg_first_innings'] + wind_impact['avg_second_innings']) / 2
    wind_impact['factor_type'] = 'Wind Speed'
    
    weather_impact.append(wind_impact)
    
    # Analyze impact of weather condition
    condition_impact = matches_weather.groupby('weather_condition').agg({
        'id': 'count',
        'team1_score': 'mean',
        'team2_score': 'mean'
    }).reset_index()
    
    condition_impact = condition_impact.rename(columns={
        'id': 'matches',
        'team1_score': 'avg_first_innings',
        'team2_score': 'avg_second_innings'
    })
    
    condition_impact['total_runs'] = (condition_impact['avg_first_innings'] + condition_impact['avg_second_innings']) / 2
    condition_impact['factor_type'] = 'Weather Condition'
    
    weather_impact.append(condition_impact)
    
    # Combine all impact analyses
    weather_impact_df = pd.concat(weather_impact, ignore_index=True)
    
    logger.info("Weather impact analysis completed")
    
    return weather_impact_df

def plot_weather_impact(weather_impact_df):
    """
    Plot the impact of weather conditions on match outcomes
    """
    logger.info("Plotting weather impact on match outcomes")
    
    if weather_impact_df is None:
        logger.error("Weather impact data not available for plotting")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plot impact of temperature on runs scored
    temp_data = weather_impact_df[weather_impact_df['factor_type'] == 'Temperature']
    if not temp_data.empty:
        plt.figure(figsize=(12, 6))
        
        x = range(len(temp_data))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], temp_data['avg_first_innings'], width, label='First Innings')
        plt.bar([i + width/2 for i in x], temp_data['avg_second_innings'], width, label='Second Innings')
        
        plt.xlabel('Temperature Range', fontsize=14)
        plt.ylabel('Average Runs', fontsize=14)
        plt.title('Impact of Temperature on Runs Scored', fontsize=16)
        plt.xticks(x, temp_data['temp_range'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(RESULTS_DIR, 'temperature_impact.png'), dpi=300)
        plt.close()
    
    # Plot impact of humidity on runs scored
    humidity_data = weather_impact_df[weather_impact_df['factor_type'] == 'Humidity']
    if not humidity_data.empty:
        plt.figure(figsize=(12, 6))
        
        x = range(len(humidity_data))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], humidity_data['avg_first_innings'], width, label='First Innings')
        plt.bar([i + width/2 for i in x], humidity_data['avg_second_innings'], width, label='Second Innings')
        
        plt.xlabel('Humidity Range', fontsize=14)
        plt.ylabel('Average Runs', fontsize=14)
        plt.title('Impact of Humidity on Runs Scored', fontsize=16)
        plt.xticks(x, humidity_data['humidity_range'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(RESULTS_DIR, 'humidity_impact.png'), dpi=300)
        plt.close()
    
    # Plot impact of weather condition on runs scored
    condition_data = weather_impact_df[weather_impact_df['factor_type'] == 'Weather Condition']
    if not condition_data.empty:
        plt.figure(figsize=(12, 6))
        
        x = range(len(condition_data))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], condition_data['avg_first_innings'], width, label='First Innings')
        plt.bar([i + width/2 for i in x], condition_data['avg_second_innings'], width, label='Second Innings')
        
        plt.xlabel('Weather Condition', fontsize=14)
        plt.ylabel('Average Runs', fontsize=14)
        plt.title('Impact of Weather Condition on Runs Scored', fontsize=16)
        plt.xticks(x, condition_data['weather_condition'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(RESULTS_DIR, 'weather_condition_impact.png'), dpi=300)
        plt.close()
    
    logger.info("Weather impact plots saved to results directory")

def save_weather_data(matches_weather, weather_impact_df):
    """
    Save weather data and impact analysis
    """
    logger.info("Saving weather data and impact analysis")
    
    if matches_weather is not None:
        # Save matches with weather data
        weather_file = os.path.join(PROCESSED_DATA_DIR, "matches_with_weather.csv")
        matches_weather.to_csv(weather_file, index=False)
        logger.info(f"Saved matches with weather data to: {weather_file}")
    
    if weather_impact_df is not None:
        # Save weather impact analysis
        impact_file = os.path.join(PROCESSED_DATA_DIR, "weather_impact_analysis.csv")
        weather_impact_df.to_csv(impact_file, index=False)
        logger.info(f"Saved weather impact analysis to: {impact_file}")
        
        # Also save to results directory for visualization
        results_file = os.path.join(RESULTS_DIR, "weather_impact_analysis.csv")
        weather_impact_df.to_csv(results_file, index=False)

def analyze_weather():
    """
    Main function to analyze weather impact on match outcomes
    """
    logger.info("Starting weather impact analysis")
    
    # Load venue and match data
    venues_df = load_venue_data()
    matches_df = load_matches_data()
    
    if venues_df is None or matches_df is None:
        logger.error("Venue or match data not available for weather analysis")
        return None, None
    
    # Generate weather data for matches
    matches_weather = generate_weather_data(venues_df, matches_df)
    
    # Analyze weather impact on match outcomes
    weather_impact_df = analyze_weather_impact(matches_weather)
    
    # Plot weather impact
    plot_weather_impact(weather_impact_df)
    
    # Save weather data and impact analysis
    save_weather_data(matches_weather, weather_impact_df)
    
    logger.info("Weather impact analysis completed")
    
    return matches_weather, weather_impact_df

if __name__ == "__main__":
    analyze_weather()