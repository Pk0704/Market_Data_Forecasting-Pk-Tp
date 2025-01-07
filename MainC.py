import fireducks.pandas as pd #just use without fireducks but add it at the end when we submit
import polars as pl
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score



#Using LGBM regressor with other simpler models
class MarketPredictor:
    def __init__(self):
        self.feature_cols = [f'feature_{i:02d}' for i in range(79)]
        #Which responder should we choose? The say 9 is the latest one so that's proabably best but I need your intuition
        self.target = 'responder_9'
        self.models = {}
        self.scaler = StandardScaler()
        
    def create_temporal_features(self, df, symbol_id):
        """Create time-based features for a specific symbol"""
        # Rolling statistics - this class and the next are from chat but it seems rlly good
        windows = [5, 10, 20]
        
        for feat in self.feature_cols:
            symbol_data = df[df['symbol_id'] == symbol_id][feat]
            
            for window in windows:
                df.loc[df['symbol_id'] == symbol_id, f'{feat}_mean_{window}'] = symbol_data.rolling(window).mean()
                df.loc[df['symbol_id'] == symbol_id, f'{feat}_std_{window}'] = symbol_data.rolling(window).std()
                
        return df
    
    def create_lag_features(self, df, symbol_id, lags=[1, 2, 3]):
        """Create lagged features for a specific symbol"""
        for feat in self.feature_cols:
            symbol_data = df[df['symbol_id'] == symbol_id][feat]
            
            for lag in lags:
                df.loc[df['symbol_id'] == symbol_id, f'{feat}_lag_{lag}'] = symbol_data.shift(lag)
                
        return df

    def prepare_features(self, df, lags_df=None):
        """Prepare features for training/prediction"""
        df = df.copy()
        
        # Add lags data if provided
        if lags_df is not None:
            for resp in [f'responder_{i}' for i in range(9)]:
                df[f'{resp}_prev'] = lags_df[resp]
        
        # Create features for each symbol
        for symbol in df['symbol_id'].unique():
            df = self.create_temporal_features(df, symbol)
            df = self.create_lag_features(df, symbol)
        
        # Drop rows with NaN (usually at the start due to rolling/lag features)
        df = df.dropna()
        
        return df
    
    #need your expertise here
    def get_feature_columns(self, df):
        """Get all feature columns including engineered ones"""
        return [col for col in df.columns 
                if col.startswith('feature_') or 
                   col.endswith(('_mean_5', '_mean_10', '_mean_20',
                               '_std_5', '_std_10', '_std_20',
                               '_lag_1', '_lag_2', '_lag_3')) or
                   col.endswith('_prev')]
    
    def train(self, train_df, lags_df=None):
        # Here we are initializing the models. No idea what numbers I should pick but from the internet these are common
        lgb = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        #use ridge
        
    def predict(self, test_df, lags_df=None):
        """Make predictions on test data"""
        # Prepare and scale the features. Then we make a prediction
        df = self.prepare_features(test_df, lags_df)
        X = self.scaler.transform(df[self.feature_cols_final])
        predictions = self.models['ensemble'].predict(X)
        return predictions
    
    def evaluate(self, y_true, y_pred, weights):
        """Calculate weighted R-squared score"""
        weighted_mse = np.sum(weights * (y_true - y_pred) ** 2)
        weighted_var = np.sum(weights * y_true ** 2)
        r2 = 1 - weighted_mse / weighted_var
        return r2

def main():
    # Example usage
    # Load data
    path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
    samples = [] 

    # Load data
    for i in range(1):
        file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
        part = pd.read_parquet(file_path)
        samples.append(part)
        
    df = pd.concat(samples, ignore_index=True)

    print(df)

if __name__ == "__main__": 
    main()


