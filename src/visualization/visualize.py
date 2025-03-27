import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(MODELS_DIR, 'results')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')

# Make sure the visualization directory exists
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_results():
    """
    Load the tournament simulation results
    """
    logger.info("Loading tournament simulation results")
    
    results = {}
    
    # Load league standings
    standings_file = os.path.join(RESULTS_DIR, 'league_standings.csv')
    if os.path.exists(standings_file):
        results['standings'] = pd.read_csv(standings_file)
        logger.info(f"Loaded league standings: {len(results['standings'])} teams")
    else:
        logger.warning(f"League standings file not found: {standings_file}")
    
    # Load tournament summary
    summary_file = os.path.join(RESULTS_DIR, 'tournament_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
        logger.info("Loaded tournament summary")
    else:
        logger.warning(f"Tournament summary file not found: {summary_file}")
    
    # Load processed data
    model_data_file = os.path.join(PROCESSED_DATA_DIR, 'model_data.csv')
    if os.path.exists(model_data_file):
        results['model_data'] = pd.read_csv(model_data_file)
        logger.info(f"Loaded model data: {len(results['model_data'])} matchups")
    else:
        logger.warning(f"Model data file not found: {model_data_file}")
    
    teams_file = os.path.join(PROCESSED_DATA_DIR, 'processed_teams.csv')
    if os.path.exists(teams_file):
        results['teams'] = pd.read_csv(teams_file)
        logger.info(f"Loaded processed teams data: {len(results['teams'])} teams")
    else:
        logger.warning(f"Processed teams file not found: {teams_file}")
    
    # Load team performance data if available
    team_performance_file = os.path.join(PROCESSED_DATA_DIR, 'processed_team_performance.csv')
    if os.path.exists(team_performance_file):
        results['team_performance'] = pd.read_csv(team_performance_file)
        logger.info(f"Loaded team performance data: {len(results['team_performance'])} teams")
    else:
        logger.warning(f"Team performance file not found: {team_performance_file}")
    
    # Load fixture predictions if available
    fixture_file = os.path.join(RESULTS_DIR, 'fixture_predictions.csv')
    if os.path.exists(fixture_file):
        results['fixtures'] = pd.read_csv(fixture_file)
        logger.info(f"Loaded fixture predictions: {len(results['fixtures'])} fixtures")
    else:
        logger.warning(f"Fixture predictions file not found: {fixture_file}")
    
    # Load comprehensive analysis if available
    comprehensive_file = os.path.join(RESULTS_DIR, 'comprehensive_analysis.json')
    if os.path.exists(comprehensive_file):
        with open(comprehensive_file, 'r') as f:
            results['comprehensive'] = json.load(f)
        logger.info("Loaded comprehensive analysis")
    else:
        logger.warning(f"Comprehensive analysis file not found: {comprehensive_file}")
    
    # Load head to head matrix if available
    h2h_file = os.path.join(RESULTS_DIR, 'head_to_head_matrix.csv')
    if os.path.exists(h2h_file):
        results['h2h_matrix'] = pd.read_csv(h2h_file, index_col=0)
        logger.info(f"Loaded head-to-head matrix")
    else:
        logger.warning(f"Head-to-head matrix file not found: {h2h_file}")
    
    # Load team analysis if available
    team_analysis_file = os.path.join(RESULTS_DIR, 'team_analysis.csv')
    if os.path.exists(team_analysis_file):
        results['team_analysis'] = pd.read_csv(team_analysis_file)
        logger.info(f"Loaded team analysis: {len(results['team_analysis'])} teams")
    else:
        logger.warning(f"Team analysis file not found: {team_analysis_file}")
    
    return results

def plot_league_standings(results):
    """
    Plot the league standings
    """
    logger.info("Plotting league standings")
    
    if 'standings' not in results or results['standings'] is None:
        logger.warning("League standings data not available")
        return
    
    standings = results['standings']
    
    # Sort by points (descending)
    standings = standings.sort_values('points', ascending=False)
    
    # Plot standings
    plt.figure(figsize=(12, 8))
    
    # Create a colormap based on whether teams made the playoffs
    if 'summary' in results and 'Playoff Teams' in results['summary']:
        playoff_teams = results['summary']['Playoff Teams']
        standings['playoff_status'] = standings['team'].apply(lambda x: 'Playoff' if x in playoff_teams else 'Non-Playoff')
        colors = {'Playoff': '#1E88E5', 'Non-Playoff': '#D81B60'}
    else:
        standings['playoff_status'] = 'Team'
        colors = {'Team': '#1E88E5'}
    
    # Create the bar plot using hue instead of palette
    ax = sns.barplot(x='team', y='points', hue='playoff_status', data=standings, palette=colors, legend=False)
    
    # Add win-loss record on top of each bar
    for i, row in enumerate(standings.itertuples()):
        wins = row.wins
        losses = row.matches - row.wins
        ax.text(i, row.points + 1, f"{wins}-{losses}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Add a horizontal line for playoff qualification
    if len(standings) >= 4:
        plt.axhline(y=standings.iloc[3]['points'] + 0.5, color='r', linestyle='--', 
                   label='Playoff Qualification Line')
    
    plt.title('IPL 2025 Predicted League Standings', fontsize=16)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Points', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'league_standings.png'), dpi=300)
    plt.close()

def plot_team_strengths(results):
    """
    Plot the team strengths
    """
    logger.info("Plotting team strengths")
    
    if 'teams' not in results or results['teams'] is None:
        logger.warning("Team data not available")
        return
    
    teams_df = results['teams'].copy()
    
    # Check if team_strength column exists
    if 'team_strength' not in teams_df.columns:
        logger.warning("Team strength data not available")
        return
    
    # Sort by team strength (descending)
    teams_df = teams_df.sort_values('team_strength', ascending=False)
    
    # Create a colormap based on whether teams made the playoffs
    if 'summary' in results and 'Playoff Teams' in results['summary']:
        playoff_teams = results['summary']['Playoff Teams']
        teams_df['playoff_status'] = teams_df['name'].apply(lambda x: 'Playoff' if x in playoff_teams else 'Non-Playoff')
        colors = {'Playoff': '#1E88E5', 'Non-Playoff': '#D81B60'}
    else:
        teams_df['playoff_status'] = 'Team'
        colors = {'Team': '#1E88E5'}
    
    # Plot team strengths
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='name', y='team_strength', hue='playoff_status', data=teams_df, palette=colors, legend=False)
    
    # Highlight the champion and runner-up if available
    if 'summary' in results:
        summary = results['summary']
        
        if 'Champion' in summary:
            champion_idx = teams_df[teams_df['name'] == summary['Champion']].index
            if len(champion_idx) > 0:
                champion_idx = champion_idx[0]
                ax.patches[champion_idx].set_edgecolor('gold')
                ax.patches[champion_idx].set_linewidth(3)
                plt.text(champion_idx, teams_df.iloc[champion_idx]['team_strength'] + 2, 
                       "Champion", ha='center', fontweight='bold', color='gold')
        
        if 'Runner-up' in summary:
            runner_up_idx = teams_df[teams_df['name'] == summary['Runner-up']].index
            if len(runner_up_idx) > 0:
                runner_up_idx = runner_up_idx[0]
                ax.patches[runner_up_idx].set_edgecolor('silver')
                ax.patches[runner_up_idx].set_linewidth(3)
                plt.text(runner_up_idx, teams_df.iloc[runner_up_idx]['team_strength'] + 2, 
                       "Runner-up", ha='center', fontweight='bold', color='silver')
    
    plt.title('IPL Teams Strength Comparison', fontsize=16)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Team Strength (0-100)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'team_strengths.png'), dpi=300)
    plt.close()

def plot_win_probability_matrix(results):
    """
    Plot a matrix of win probabilities for all team matchups
    """
    logger.info("Plotting win probability matrix")
    
    if 'model_data' not in results or results['model_data'] is None:
        logger.warning("Model data not available")
        return
    
    model_data = results['model_data']
    
    # Get all teams
    teams = sorted(model_data['team1'].unique())
    num_teams = len(teams)
    
    # Create a DataFrame to store win probabilities
    win_prob_matrix = pd.DataFrame(index=teams, columns=teams)
    
    # Fill the matrix
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i == j:
                # Teams don't play against themselves
                win_prob_matrix.loc[team1, team2] = 0.5
            else:
                # Find the matchup in the data
                matchup = model_data[(model_data['team1'] == team1) & (model_data['team2'] == team2)]
                
                if len(matchup) > 0:
                    # Get the win probability for team1 against team2
                    win_prob = matchup['team1_win_probability'].values[0] / 100
                    win_prob_matrix.loc[team1, team2] = win_prob
                else:
                    # Try the reverse matchup
                    matchup = model_data[(model_data['team1'] == team2) & (model_data['team2'] == team1)]
                    
                    if len(matchup) > 0:
                        # Get the win probability for team2 against team1 (reverse it)
                        win_prob = 1 - (matchup['team1_win_probability'].values[0] / 100)
                        win_prob_matrix.loc[team1, team2] = win_prob
                    else:
                        # No data for this matchup
                        win_prob_matrix.loc[team1, team2] = 0.5
    
    # Convert all values to float to ensure no object dtype issues
    win_prob_matrix = win_prob_matrix.astype(float)
    
    # Plot the win probability matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(win_prob_matrix, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Win Probability'},
               vmin=0, vmax=1, fmt='.2f')
    plt.title('Head-to-Head Win Probability Matrix', fontsize=16)
    plt.xlabel('Team 2', fontsize=14)
    plt.ylabel('Team 1', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'win_probability_matrix.png'), dpi=300)
    plt.close()

def plot_playoff_bracket(results):
    """
    Plot the playoff bracket
    """
    logger.info("Plotting playoff bracket")
    
    if 'summary' not in results or results['summary'] is None:
        logger.warning("Tournament summary not available")
        return
    
    summary = results['summary']
    
    # Check if we have all the playoff data
    if not all(key in summary for key in ['Playoff Teams', 'Qualifier 1 Winner', 
                                          'Eliminator Winner', 'Qualifier 2 Winner', 
                                          'Champion', 'Runner-up']):
        logger.warning("Complete playoff data not available")
        return
    
    # Create a figure with custom playoff bracket layout
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Hide axes
    ax.axis('off')
    
    # Playoff teams
    playoff_teams = summary['Playoff Teams']
    if len(playoff_teams) < 4:
        logger.warning("Not enough playoff teams")
        return
    
    # Qualifier 1: 1st vs 2nd
    team1, team2 = playoff_teams[0], playoff_teams[1]
    q1_winner = summary['Qualifier 1 Winner']
    q1_loser = team1 if q1_winner == team2 else team2
    
    # Eliminator: 3rd vs 4th
    team3, team4 = playoff_teams[2], playoff_teams[3]
    eliminator_winner = summary['Eliminator Winner']
    
    # Qualifier 2: Q1 Loser vs Eliminator Winner
    q2_winner = summary['Qualifier 2 Winner']
    
    # Final
    champion = summary['Champion']
    runner_up = summary['Runner-up']
    
    # Define positions for the bracket
    positions = {
        'q1_team1': (0.2, 0.8),
        'q1_team2': (0.2, 0.7),
        'q1_winner': (0.4, 0.75),
        'eliminator_team1': (0.2, 0.3),
        'eliminator_team2': (0.2, 0.2),
        'eliminator_winner': (0.4, 0.25),
        'q2_team1': (0.6, 0.3),
        'q2_team2': (0.6, 0.2),
        'q2_winner': (0.8, 0.25),
        'final_team1': (0.8, 0.75),
        'final_team2': (0.8, 0.25),
        'champion': (1.0, 0.5)
    }
    
    # Function to highlight winner
    def draw_match(ax, team1_pos, team2_pos, winner_pos, team1, team2, winner):
        # Draw lines
        ax.plot([team1_pos[0], winner_pos[0]], [team1_pos[1], winner_pos[1]], 'k-', alpha=0.5)
        ax.plot([team2_pos[0], winner_pos[0]], [team2_pos[1], winner_pos[1]], 'k-', alpha=0.5)
        
        # Draw team boxes
        team1_color = 'lightgreen' if team1 == winner else 'white'
        team2_color = 'lightgreen' if team2 == winner else 'white'
        
        ax.text(team1_pos[0], team1_pos[1], team1, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor=team1_color, alpha=0.8, pad=0.5),
               fontsize=12)
        
        ax.text(team2_pos[0], team2_pos[1], team2, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor=team2_color, alpha=0.8, pad=0.5),
               fontsize=12)
        
        # Draw winner at the next position
        ax.text(winner_pos[0], winner_pos[1], winner, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5),
               fontsize=12, fontweight='bold')
    
    # Draw Qualifier 1
    draw_match(ax, positions['q1_team1'], positions['q1_team2'], positions['q1_winner'], 
              team1, team2, q1_winner)
    
    # Draw Eliminator
    draw_match(ax, positions['eliminator_team1'], positions['eliminator_team2'], 
              positions['eliminator_winner'], team3, team4, eliminator_winner)
    
    # Draw Qualifier 2
    draw_match(ax, positions['q1_winner'], positions['eliminator_winner'], 
              positions['q2_winner'], q1_loser, eliminator_winner, q2_winner)
    
    # Draw Final
    draw_match(ax, positions['final_team1'], positions['final_team2'], 
              positions['champion'], q1_winner, q2_winner, champion)
    
    # Add labels for each match
    plt.text(0.2, 0.85, "Qualifier 1", ha='center', fontsize=14, fontweight='bold')
    plt.text(0.2, 0.35, "Eliminator", ha='center', fontsize=14, fontweight='bold')
    plt.text(0.6, 0.35, "Qualifier 2", ha='center', fontsize=14, fontweight='bold')
    plt.text(0.8, 0.85, "Final", ha='center', fontsize=14, fontweight='bold')
    
    # Draw the champion trophy
    plt.text(1.0, 0.5, "🏆", ha='center', va='center', fontsize=40)
    plt.text(1.0, 0.42, champion, ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.title('IPL 2025 Playoff Bracket Prediction', fontsize=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'playoff_bracket.png'), dpi=300)
    plt.close()

def plot_champion_probability(results):
    """
    Plot the probability of each team winning the tournament
    """
    logger.info("Plotting champion probability")
    
    if 'standings' not in results or results['standings'] is None:
        logger.warning("League standings data not available")
        return
    
    standings = results['standings']
    
    # We'll use points as a proxy for championship probability
    total_points = standings['points'].sum()
    if total_points == 0:
        logger.warning("No points data available")
        return
    
    # Calculate championship probability based on points
    standings['champ_probability'] = standings['points'] / total_points * 100
    
    # Sort by probability (descending)
    standings = standings.sort_values('champ_probability', ascending=False)
    
    # Plot championship probability
    plt.figure(figsize=(12, 8))
    
    # Create a colormap based on whether teams made the playoffs
    if 'summary' in results and 'Playoff Teams' in results['summary']:
        playoff_teams = results['summary']['Playoff Teams']
        colors = ['#1E88E5' if team in playoff_teams else '#D81B60' for team in standings['team']]
    else:
        colors = sns.color_palette('viridis', len(standings))
    
    # Create the bar plot
    ax = sns.barplot(x='team', y='champ_probability', data=standings, palette=colors)
    
    # Highlight the actual champion if available
    if 'summary' in results and 'Champion' in results['summary']:
        champion = results['summary']['Champion']
        champion_idx = standings[standings['team'] == champion].index
        if len(champion_idx) > 0:
            champion_idx = champion_idx[0]
            ax.patches[champion_idx].set_edgecolor('gold')
            ax.patches[champion_idx].set_linewidth(3)
            plt.text(champion_idx, standings.iloc[champion_idx]['champ_probability'] + 1, 
                   "Champion", ha='center', fontweight='bold', color='gold')
    
    plt.title('IPL 2025 Championship Probability', fontsize=16)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Championship Probability (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'championship_probability.png'), dpi=300)
    plt.close()

def create_summary_dashboard(results):
    """
    Create a summary dashboard of the IPL prediction results
    """
    logger.info("Creating summary dashboard")
    
    if 'summary' not in results or results['summary'] is None:
        logger.warning("Tournament summary not available")
        return
    
    summary = results['summary']
    
    # Create a dashboard figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: League Standings (top left)
    if 'standings' in results and results['standings'] is not None:
        standings = results['standings'].sort_values('points', ascending=False)
        
        # Create a colormap based on whether teams made the playoffs
        if 'Playoff Teams' in summary:
            playoff_teams = summary['Playoff Teams']
            colors = ['#1E88E5' if team in playoff_teams else '#D81B60' for team in standings['team']]
        else:
            colors = ['#1E88E5'] * len(standings)
        
        # Create the bar plot
        ax = axs[0, 0]
        sns.barplot(x='team', y='points', data=standings, palette=colors, ax=ax)
        
        # Add win-loss record on top of each bar
        for i, row in enumerate(standings.itertuples()):
            wins = row.wins
            losses = row.matches - row.wins
            ax.text(i, row.points + 1, f"{wins}-{losses}", 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('League Standings', fontsize=16)
        ax.set_xlabel('Team', fontsize=14)
        ax.set_ylabel('Points', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    # Plot 2: Team Strengths (top right)
    if 'teams' in results and results['teams'] is not None:
        teams_df = results['teams']
        
        if 'team_strength' in teams_df.columns:
            teams_df = teams_df.sort_values('team_strength', ascending=False)
            
            # Create a colormap based on whether teams made the playoffs
            if 'Playoff Teams' in summary:
                playoff_teams = summary['Playoff Teams']
                colors = ['#1E88E5' if team in playoff_teams else '#D81B60' for team in teams_df['name']]
            else:
                colors = sns.color_palette('viridis', len(teams_df))
            
            ax = axs[0, 1]
            sns.barplot(x='name', y='team_strength', data=teams_df, palette=colors, ax=ax)
            ax.set_title('Team Strengths', fontsize=16)
            ax.set_xlabel('Team', fontsize=14)
            ax.set_ylabel('Team Strength (0-100)', fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    # Plot 3: Championship Probability (bottom left)
    if 'standings' in results and results['standings'] is not None:
        standings = results['standings'].copy()
        
        # We'll use points as a proxy for championship probability
        total_points = standings['points'].sum()
        if total_points > 0:
            standings['champ_probability'] = standings['points'] / total_points * 100
            standings = standings.sort_values('champ_probability', ascending=False)
            
            ax = axs[1, 0]
            sns.barplot(x='team', y='champ_probability', data=standings, palette=colors, ax=ax)
            
            # Highlight the actual champion if available
            if 'Champion' in summary:
                champion = summary['Champion']
                champion_idx = standings[standings['team'] == champion].index
                if len(champion_idx) > 0:
                    champion_idx = champion_idx[0]
                    ax.patches[champion_idx].set_edgecolor('gold')
                    ax.patches[champion_idx].set_linewidth(3)
            
            ax.set_title('Championship Probability', fontsize=16)
            ax.set_xlabel('Team', fontsize=14)
            ax.set_ylabel('Probability (%)', fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
    
    # Plot 4: Playoff Results (bottom right)
    ax = axs[1, 1]
    ax.axis('off')
    
    # Display playoff results as text
    playoff_text = "Playoff Results:\n\n"
    
    if 'Playoff Teams' in summary:
        playoff_text += f"Playoff Teams: {', '.join(summary['Playoff Teams'])}\n\n"
    
    if 'Qualifier 1 Winner' in summary:
        playoff_text += f"Qualifier 1 Winner: {summary['Qualifier 1 Winner']}\n"
    
    if 'Eliminator Winner' in summary:
        playoff_text += f"Eliminator Winner: {summary['Eliminator Winner']}\n"
    
    if 'Qualifier 2 Winner' in summary:
        playoff_text += f"Qualifier 2 Winner: {summary['Qualifier 2 Winner']}\n\n"
    
    if 'Champion' in summary:
        playoff_text += f"Champion: {summary['Champion']}\n"
    
    if 'Runner-up' in summary:
        playoff_text += f"Runner-up: {summary['Runner-up']}\n"
    
    ax.text(0.1, 0.5, playoff_text, fontsize=16, va='center', linespacing=1.5)
    ax.set_title('Playoff Results', fontsize=16)
    
    # Main title
    fig.suptitle('IPL 2025 Tournament Prediction Summary', fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'summary_dashboard.png'), dpi=300)
    plt.close()

def plot_team_analysis(results):
    """
    Plot team analysis visualizations
    """
    logger.info("Plotting team analysis")
    
    if 'team_analysis' not in results or results['team_analysis'] is None:
        logger.warning("Team analysis data not available")
        return
    
    team_analysis = results['team_analysis']
    
    # 1. Create a scatter plot of batting vs bowling
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        team_analysis['batting_avg'], 
        team_analysis['bowling_economy'],
        s=100, 
        c=team_analysis['overall_rating'],
        cmap='viridis',
        alpha=0.8
    )
    
    # Add team labels to scatter points
    for i, row in team_analysis.iterrows():
        plt.annotate(
            row['team'],
            (row['batting_avg'], row['bowling_economy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add a colorbar to indicate overall rating
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overall Rating', fontsize=12)
    
    # Better labels and styling
    plt.title('Team Performance: Batting vs Bowling', fontsize=16)
    plt.xlabel('Batting Average', fontsize=14)
    plt.ylabel('Bowling Economy Rate (lower is better)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'team_performance_scatter.png'), dpi=300)
    plt.close()
    
    # 2. Strength and weakness visualization
    # Prepare data for strengths and weaknesses analysis
    plt.figure(figsize=(14, 8))
    
    # Sort by overall rating
    team_analysis = team_analysis.sort_values('overall_rating', ascending=False)
    
    # Create subplot grid
    n_teams = len(team_analysis)
    cols = 3
    rows = (n_teams + cols - 1) // cols
    
    for i, (_, team) in enumerate(team_analysis.iterrows()):
        ax = plt.subplot(rows, cols, i+1)
        
        # Extract strengths and weaknesses
        strengths = team['strengths'].split(', ') if not pd.isna(team['strengths']) else []
        weaknesses = team['weaknesses'].split(', ') if not pd.isna(team['weaknesses']) else []
        
        # Combine the data
        categories = strengths + weaknesses
        values = [1] * len(strengths) + [-1] * len(weaknesses)
        
        # Create colors
        colors = ['green'] * len(strengths) + ['red'] * len(weaknesses)
        
        # Create the bar chart
        bars = ax.bar(range(len(categories)), values, color=colors)
        
        # Add labels
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        
        # Set y limits
        ax.set_ylim(-1.5, 1.5)
        
        # Remove y ticks
        ax.set_yticks([])
        
        # Add title
        ax.set_title(f"{team['team']} (Rating: {team['overall_rating']:.1f})", fontsize=10)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels inside bars
        for j, bar in enumerate(bars):
            label = "Strength" if values[j] > 0 else "Weakness"
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() * 0.5 * np.sign(values[j]),
                categories[j],
                ha='center',
                va='center',
                fontsize=8,
                color='white',
                fontweight='bold'
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'team_strengths_weaknesses.png'), dpi=300)
    plt.close()
    
    # 3. Overall team ratings visualization
    plt.figure(figsize=(12, 8))
    
    # Create a color palette based on team tiers if available
    if 'tier' in team_analysis.columns and not team_analysis['tier'].isna().all():
        # Get unique tiers
        tiers = team_analysis['tier'].unique()
        tier_colors = dict(zip(tiers, sns.color_palette('viridis', len(tiers))))
        
        # Add tier as a column for hue
        team_analysis['tier_label'] = team_analysis['tier'].astype(str)
        
        # Create the bar chart with hue
        ax = sns.barplot(x='team', y='overall_rating', hue='tier_label', data=team_analysis, palette=tier_colors, legend=False)
    else:
        # Add a dummy column for hue
        team_analysis['team_group'] = 'All Teams'
        
        # Use standard color palette
        colors = sns.color_palette('viridis', len(team_analysis))
        team_colors = dict(zip(team_analysis['team_group'].unique(), [colors]))
        
        # Create the bar chart
        ax = sns.barplot(x='team', y='overall_rating', hue='team_group', data=team_analysis, palette=team_colors, legend=False)
    
    # Add value labels
    for i, v in enumerate(team_analysis['overall_rating']):
        ax.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=10)
    
    # Add tier labels if available
    if 'tier' in team_analysis.columns and not team_analysis['tier'].isna().all():
        for i, (_, team) in enumerate(team_analysis.iterrows()):
            if not pd.isna(team['tier']):
                ax.text(i, 5, f"Tier: {team['tier']}", ha='center', fontsize=9, color='black', alpha=0.7)
    
    # Add styling
    plt.title('Team Overall Ratings', fontsize=16)
    plt.xlabel('Team', fontsize=14)
    plt.ylabel('Overall Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'team_overall_ratings.png'), dpi=300)
    plt.close()
    
    logger.info("Completed team analysis visualizations")

def plot_fixture_predictions(results):
    """
    Plot predictions for upcoming fixtures
    """
    logger.info("Plotting fixture predictions")
    
    if 'fixtures' not in results or results['fixtures'] is None:
        logger.warning("Fixture data not available")
        return
    
    fixtures = results['fixtures']
    
    # Filter to only upcoming (not completed) fixtures
    upcoming = fixtures[fixtures['completed'] == False].copy()
    
    if len(upcoming) == 0:
        logger.warning("No upcoming fixtures to visualize")
        return
    
    # Ensure prediction columns exist
    if 'prediction' not in upcoming.columns or 'prediction_confidence' not in upcoming.columns:
        logger.warning("Prediction data not available for fixtures")
        return
    
    # Sort fixtures by date if available
    if 'date' in upcoming.columns:
        upcoming['date'] = pd.to_datetime(upcoming['date'], errors='coerce')
        upcoming = upcoming.sort_values('date')
    
    # Create a better display name for each fixture
    upcoming['match_display'] = upcoming.apply(
        lambda x: f"{x['team1']} vs {x['team2']}" + 
                 (f" ({x['date'].strftime('%Y-%m-%d')})" if 'date' in upcoming.columns and pd.notna(x['date']) else ""),
        axis=1
    )
    
    # Create colors based on prediction
    upcoming['color'] = upcoming.apply(
        lambda x: 'blue' if x['prediction'] == x['team1'] else 'red',
        axis=1
    )
    
    # Calculate win probabilities for visualization
    upcoming['team1_win_prob'] = upcoming.apply(
        lambda x: x['prediction_confidence'] if x['prediction'] == x['team1'] else 100 - x['prediction_confidence'],
        axis=1
    )
    upcoming['team2_win_prob'] = 100 - upcoming['team1_win_prob']
    
    # Create a horizontal stacked bar chart
    plt.figure(figsize=(12, max(6, len(upcoming) * 0.4)))
    
    # Plot bars
    y_pos = range(len(upcoming))
    
    # Team 1 bars
    plt.barh(
        y_pos, 
        upcoming['team1_win_prob'], 
        color='blue', 
        alpha=0.7,
        label=None
    )
    
    # Team 2 bars
    plt.barh(
        y_pos, 
        upcoming['team2_win_prob'], 
        left=upcoming['team1_win_prob'], 
        color='red', 
        alpha=0.7,
        label=None
    )
    
    # Add labels for team names
    for i, row in enumerate(upcoming.iterrows()):
        row = row[1]  # Get the row data
        
        # Team 1 label
        plt.text(
            2, 
            i, 
            row['team1'], 
            color='white', 
            fontweight='bold', 
            va='center',
            fontsize=9
        )
        
        # Team 2 label
        plt.text(
            98, 
            i, 
            row['team2'], 
            color='white', 
            fontweight='bold', 
            va='center',
            ha='right',
            fontsize=9
        )
        
        # Add win probability percentages
        plt.text(
            row['team1_win_prob'] / 2, 
            i, 
            f"{row['team1_win_prob']:.1f}%", 
            color='white', 
            fontweight='bold', 
            va='center', 
            ha='center',
            fontsize=9
        )
        
        plt.text(
            row['team1_win_prob'] + row['team2_win_prob'] / 2, 
            i, 
            f"{row['team2_win_prob']:.1f}%", 
            color='white', 
            fontweight='bold', 
            va='center', 
            ha='center',
            fontsize=9
        )
    
    # Add match details on the y-axis
    plt.yticks(y_pos, upcoming['match_display'])
    
    # Add styling
    plt.title('Predicted Match Win Probabilities', fontsize=16)
    plt.xlabel('Win Probability (%)', fontsize=14)
    plt.xlim(0, 100)
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Team 1 Win Probability'),
        Patch(facecolor='red', alpha=0.7, label='Team 2 Win Probability')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'fixture_predictions.png'), dpi=300)
    plt.close()
    
    logger.info("Completed fixture predictions visualization")

def create_head_to_head_heatmap(results):
    """
    Create a heatmap of head-to-head win probabilities
    """
    logger.info("Creating head-to-head heatmap")
    
    if 'h2h_matrix' not in results or results['h2h_matrix'] is None:
        logger.warning("Head-to-head matrix not available")
        return
    
    h2h_matrix = results['h2h_matrix']
    
    # Create the heatmap
    plt.figure(figsize=(14, 12))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(h2h_matrix, dtype=bool))
    
    # Plot the heatmap
    sns.heatmap(
        h2h_matrix, 
        annot=True, 
        fmt='.1f', 
        cmap='YlGnBu', 
        mask=mask,
        vmin=0, 
        vmax=100,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Win Probability (%)'}
    )
    
    # Add styling
    plt.title('Head-to-Head Win Probability Matrix (%)', fontsize=16)
    plt.xlabel('Team (as Opponent)', fontsize=14)
    plt.ylabel('Team', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'head_to_head_heatmap.png'), dpi=300)
    plt.close()
    
    logger.info("Completed head-to-head heatmap visualization")

def main():
    """
    Main function to generate all visualizations
    """
    logger.info("Starting visualization generation")
    
    # Load results
    results = load_results()
    
    # Create league standings visualization
    plot_league_standings(results)
    
    # Create team strengths visualization
    plot_team_strengths(results)
    
    # Create team analysis visualizations
    plot_team_analysis(results)
    
    # Create fixture predictions visualization 
    plot_fixture_predictions(results)
    
    # Create head-to-head heatmap
    create_head_to_head_heatmap(results)
    
    # Create win probability matrix visualization
    plot_win_probability_matrix(results)
    
    logger.info("Visualization generation completed successfully")
    logger.info(f"All visualizations saved to {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main() 