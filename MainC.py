import pandas as pd #just use without fireducks but add it at the end when we submit
import polars as pl
from scipy.cluster import hierarchy
from scipy.stats import randint, uniform  # needed for RandomizedSearchCV
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



class CorrelationAnalyzer:
    def __init__(self, correlation_threshold=0.8):
        """
        Initialize CorrelationAnalyzer
        
        Parameters:
        correlation_threshold (float): Threshold for feature grouping
        """
        self.correlation_threshold = correlation_threshold
        self.feature_groups = {}
        self.selected_features = []
        
    def calculate_correlations(self, df, feature_cols):
        """
        Calculate Pearson correlation matrix for features
        """
        return df[feature_cols].corr()
    
    def find_feature_groups(self, df, feature_cols):
        """
        Find groups of highly correlated features using hierarchical clustering
        """
        corr = self.calculate_correlations(df, feature_cols)
        
        distance_matrix = 1 - np.abs(corr)
   
        linkage = hierarchy.linkage(distance_matrix, method='complete')
        clusters = hierarchy.fcluster(linkage, self.correlation_threshold, criterion='distance')
        
        # Group features
        self.feature_groups = {}
        for feature, cluster_id in zip(feature_cols, clusters):
            if cluster_id not in self.feature_groups:
                self.feature_groups[cluster_id] = []
            self.feature_groups[cluster_id].append(feature)
            
        return self.feature_groups
    
    def select_representative_features(self, df, target):
        """
        Select representative features from each group based on correlation with target
        """
        self.selected_features = []
        
        for group in self.feature_groups.values():
            # Calculate correlation with target for each feature in group
            correlations = []
            for feature in group:
                correlation = abs(df[feature].corr(df[target]))
                correlations.append((feature, correlation))
            
            # Select feature with highest correlation with target
            best_feature = max(correlations, key=lambda x: x[1])[0]
            self.selected_features.append(best_feature)
            
        return self.selected_features
    
    def plot_correlation_matrix(self, df, feature_cols, figsize=(12, 8)):
        """
        Plot correlation matrix heatmap
        """
        plt.figure(figsize=figsize)
        corr = self.calculate_correlations(df, feature_cols)
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()


from sklearn.model_selection import RandomizedSearchCV



#Using LGBM regressor with other simpler models
class MarketPredictor:
    """
    Class for market prediction using a LightGBM model.

    Attributes:
        feature_cols (list): List of feature column names.
        target (str): Name of the target variable.
        models (dict): Dictionary to store trained models.
        scaler (StandardScaler): StandardScaler object for feature scaling.
    """

    #initialize default parameters
    def __init__(self):
        self.feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.target = 'responder_6' 
        self.models = {}
        self.scaler = StandardScaler()

    def create_temporal_features(self, df, symbol_id):
        """Creates time-based features with careful handling of NaN values"""
        windows = [5, 10, 20]
        temp_df = df.copy()
        
        for feat in self.feature_cols:
            symbol_data = temp_df[temp_df['symbol_id'] == symbol_id][feat]
            for window in windows:
                # Use min_periods parameter to allow partial windows
                temp_df.loc[temp_df['symbol_id'] == symbol_id, f'{feat}_mean_{window}'] = (
                    symbol_data.rolling(window, min_periods=1).mean()
                )
                temp_df.loc[temp_df['symbol_id'] == symbol_id, f'{feat}_std_{window}'] = (
                    symbol_data.rolling(window, min_periods=1).std()
                )
        
        return temp_df

    def create_lag_features(self, df, symbol_id, lags=[1, 2, 3]):
        """Creates lagged features with careful handling of NaN values"""
        temp_df = df.copy()
        
        for feat in self.feature_cols:
            symbol_data = temp_df[temp_df['symbol_id'] == symbol_id][feat]
            for lag in lags:
                temp_df.loc[temp_df['symbol_id'] == symbol_id, f'{feat}_lag_{lag}'] = (
                    symbol_data.shift(lag)
                )
        
        return temp_df

    def prepare_features(self, df, lags_df=None):
        """Prepares features with NaN handling"""
        print(f"Initial DataFrame shape: {df.shape}")
        temp_df = df.copy()
        
        # Add lagged responder values if available
        if lags_df is not None:
            print(f"Lags DataFrame shape: {lags_df.shape}")
            responders_before = [f'responder_{i}' for i in range(6)]
            responders_after = [f'responder_{i}' for i in range(7, 9)]
            
            for resp in responders_before + responders_after:
                if resp in lags_df.columns:
                    temp_df[f'{resp}_prev'] = lags_df[resp]
        
        # Create features for each symbol
        for symbol in temp_df['symbol_id'].unique():
            temp_df = self.create_temporal_features(temp_df, symbol)
            temp_df = self.create_lag_features(temp_df, symbol)
        
        print(f"Shape after feature engineering: {temp_df.shape}")
        
        # Handle missing values more carefully
        feature_cols = self.get_feature_columns(temp_df)
        
        # Fill NaN values with appropriate methods
        for col in feature_cols:
            if col.endswith(('_mean_5', '_mean_10', '_mean_20')):
                temp_df[col].fillna(method='ffill', inplace=True)
                temp_df[col].fillna(method='bfill', inplace=True)
            elif col.endswith(('_std_5', '_std_10', '_std_20')):
                temp_df[col].fillna(0, inplace=True)
            elif col.endswith(('_lag_1', '_lag_2', '_lag_3')):
                temp_df[col].fillna(method='ffill', inplace=True)
                temp_df[col].fillna(0, inplace=True)
        
        # Drop any remaining rows with NaN values
        temp_df = temp_df.dropna(subset=[self.target])
        print(f"Final shape after handling NaN: {temp_df.shape}")
        
        return temp_df

    def get_feature_columns(self, df):
        """
        Gets all feature column names, including engineered features.

        Args:
            df (pd.DataFrame): DataFrame containing the features.

        Returns:
            list: List of feature column names.
        """
        return [col for col in df.columns
                   if col.startswith('feature_') or
                      col.endswith(('_mean_5', '_mean_10', '_mean_20',
                                   '_std_5', '_std_10', '_std_20',
                                   '_lag_1', '_lag_2', '_lag_3')) or
                      col.endswith('_prev')]

    def train(self, train_df, lags_df=None):
        """Modified train method with better error handling"""
        df = self.prepare_features(train_df, lags_df)
        
        if df.empty:
            raise ValueError("DataFrame is empty after preparation!")
        
        # Use correlation analyzer
        correlation_analyzer = CorrelationAnalyzer()
        feature_cols = self.get_feature_columns(df)
        
        if not feature_cols:
            raise ValueError("No feature columns found!")
            
        correlation_analyzer.find_feature_groups(df, feature_cols)
        selected_features = correlation_analyzer.select_representative_features(df, self.target)
        self.feature_cols_final = selected_features

        param_distributions = {
            'n_estimators': randint(100, 3000),
            'learning_rate': uniform(0.001, 0.1),
            'max_depth': randint(3, 15),
            'num_leaves': randint(20, 100),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(1, 50),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2)
        }

        base_model = LGBMRegressor(random_state=42)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        X = df[self.feature_cols_final]
        y = df[self.target]

        if X.empty or y.empty:
            raise ValueError("Feature or target data is empty!")

        random_search.fit(X, y)
        return random_search.best_estimator_


    def predict(self, test_df, lags_df=None):
        """
        Makes predictions on test data.

        Args:
            test_df (pd.DataFrame): DataFrame containing test data.
            lags_df (pd.DataFrame, optional): DataFrame containing lagged responder values 
                                              for test data. Defaults to None.

        Returns:
            np.ndarray: Array of predictions.
        """
        df = self.prepare_features(test_df, lags_df)
        X = self.scaler.transform(df[self.feature_cols_final])
        predictions = self.models['ensemble'].predict(X)
        return predictions

    def evaluate(self, y_true, y_pred, weights):
        """
        Calculates the weighted R-squared score.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
            weights (np.ndarray): Array of weights for each prediction.

        Returns:
            float: Weighted R-squared score.
        """
        weighted_mse = np.sum(weights * (y_true - y_pred) ** 2)
        weighted_var = np.sum(weights * y_true ** 2)
        r2 = 1 - weighted_mse / weighted_var
        return r2
        
def main():
    
    # file path
    path = "/Users/peterk/Downloads/JS Folder/jane-street-real-time-market-data-forecasting"
    samples = [] 

    # Load data
    for i in range(1):
        file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
        try:
            part = pd.read_parquet(file_path)
            print(f"Loaded partition {i} with shape: {part.shape}")  # Debug print
            samples.append(part)
        except Exception as e:
            print(f"Error loading partition {i}: {str(e)}")

    #transform into single dataframe
    df = pd.concat(samples, ignore_index=True)
    print(f"Combined DataFrame shape: {df.shape}")  # Debug print

    #initialize Market Predictor
    predictor = MarketPredictor()
    
    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Create lags dataframe for training with appropriate responder columns
    # Include all responders except 6 as features, plus responder_6 itself
    responder_cols = ([f'responder_{i}' for i in range(6)] + 
                     [f'responder_{i}' for i in range(7, 9)])
    lags_df = train_df[responder_cols].shift(1)
    print(f"Lags DataFrame shape before dropna: {lags_df.shape}")  # Debug print
    lags_df = lags_df.dropna()
    print(f"Lags DataFrame shape after dropna: {lags_df.shape}")
    
    # Prepare features for training
    train_prepared = predictor.prepare_features(train_df, lags_df)
    if train_prepared.empty:
        raise ValueError("No data left after feature preparation!")
    
    
    # Get feature columns
    feature_cols = predictor.get_feature_columns(train_prepared)
    print(f"Number of features: {len(feature_cols)}")  # Debug print
    predictor.feature_cols_final = feature_cols
    
    # Scale the features
    X_train = train_prepared[feature_cols]
    if X_train.empty:
        raise ValueError("No features to scale!")
    predictor.scaler.fit_transform(X_train)
    
    # Train the model
    best_model = predictor.train(train_prepared, lags_df)
    predictor.models['ensemble'] = best_model
    
    # Prepare test data with same responder columns
    test_lags_df = test_df[responder_cols].shift(1)
    test_lags_df = test_lags_df.dropna()
    
    # Make predictions
    predictions = predictor.predict(test_df, test_lags_df)
    
    # Calculate weights for evaluation (equal weights in this case)
    weights = np.ones(len(predictions)) / len(predictions)
    
    # Model Evaluation with R^2
    r2_score = predictor.evaluate(
        test_df[predictor.target].iloc[1:],  # Shift by 1 to align with predictions
        predictions,
        weights
    )
    
    print(f"Model R-squared score: {r2_score}")
    
if __name__ == "__main__": 
    main()
