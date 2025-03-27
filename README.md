# IPL 2025 Prediction Model

This project predicts the most likely winner of the 2025 IPL tournament based on historical data and team performance metrics.

## Project Structure

- `data/`: Contains raw and processed IPL data
  - Historical match data
  - Player statistics
  - Team compositions
  - Venue information
- `models/`: Saved prediction models
  - Trained ML models
  - Model evaluation metrics
- `notebooks/`: Jupyter notebooks for analysis and visualization
  - Data exploration
  - Feature engineering
  - Model development
- `src/`: Source code
  - `data/`: Scripts for data collection and processing
  - `features/`: Feature engineering code
  - `models/`: Model training and evaluation
  - `visualization/`: Code for generating visualizations
- `visualizations/`: Generated plots and charts
- `main.py`: Main application entry point

## Features

- Real-time match prediction
- Team performance analysis
- Player impact assessment
- Venue-based predictions
- Historical trend analysis
- Interactive visualizations
- API endpoints for predictions

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ipl-prediction.git
   cd ipl-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data collection scripts:
   ```
   python src/data/collect_data.py
   ```

4. Start the application:
   ```
   python main.py
   ```

## Modeling Approach

This project uses several machine learning models including:
- Random Forest
- XGBoost
- Neural Networks

to predict IPL match outcomes and tournament winners based on:
- Historical performance
- Current team composition
- Player statistics
- Venue statistics
- Recent form
- Head-to-head records
- Home/Away advantage
- Weather conditions

## Results

The model provides:
- Win probability for each team
- Key factors influencing team performance
- Strengths and weaknesses analysis
- Player contribution metrics
- Venue-specific predictions
- Tournament progression simulation

## API Endpoints

- `/predict`: Get match predictions
- `/team-analysis`: Get detailed team analysis
- `/player-stats`: Get player statistics
- `/venue-analysis`: Get venue-specific insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
