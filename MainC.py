import pandas as pd
import polars as pl
from scipy.cluster import hierarchy
from scipy.stats import randint, uniform
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

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
        Calculate Pearson correlation matrix for features with NaN handling
        """
        print("Calculating correlations...")
        # Remove columns with all NaN values
        valid_cols = []
        for col in feature_cols:
            if not df[col].isnull().all():
                valid_cols.append(col)
        
        if not valid_cols:
            raise ValueError("No valid features found after removing NaN columns")
        
        print(f"Number of valid columns: {len(valid_cols)}")
        
        # Calculate correlations
        corr = df[valid_cols].corr()
        
        # Fill NaN values with 0 correlation
        corr = corr.fillna(0)
        
        print(f"Correlation matrix shape: {corr.shape}")
        return corr
    
    def find_feature_groups(self, df, feature_cols):
        """
        Find groups of highly correlated features using hierarchical clustering
        with proper handling of non-finite values
        """
        try:
            corr = self.calculate_correlations(df, feature_cols)
            
            # Plot correlation matrix before clustering
            self.plot_correlation_matrix(df, feature_cols)
            
            # Convert correlation matrix to distance matrix
            distance_matrix = 1 - np.abs(corr)
            
            # Convert distance matrix to condensed form
            # Get upper triangular part excluding diagonal
            tri_upper = distance_matrix.values[np.triu_indices(n=distance_matrix.shape[0], k=1)]
            
            # Check for any remaining non-finite values
            if not np.all(np.isfinite(tri_upper)):
                # Replace any remaining non-finite values with 1 (maximum distance)
                tri_upper = np.nan_to_num(tri_upper, nan=1.0, posinf=1.0, neginf=1.0)
            
            # Perform hierarchical clustering
            linkage = hierarchy.linkage(tri_upper, method='complete')
            clusters = hierarchy.fcluster(linkage, self.correlation_threshold, criterion='distance')
            
            # Plot dendrogram
            plt.figure(figsize=(12, 8))
            hierarchy.dendrogram(linkage)
            plt.title('Feature Clustering Dendrogram')
            plt.xlabel('Feature Index')
            plt.ylabel('Distance')
            plt.show()
            
            # Group features
            self.feature_groups = {}
            valid_features = corr.columns  # Use only the valid features
            for feature, cluster_id in zip(valid_features, clusters):
                if cluster_id not in self.feature_groups:
                    self.feature_groups[cluster_id] = []
                self.feature_groups[cluster_id].append(feature)
                
            # Plot feature groups
            self.plot_feature_groups()
            
            return self.feature_groups
            
        except Exception as e:
            print(f"Error in feature grouping: {str(e)}")
            # Return single group with all features if clustering fails
            self.feature_groups = {1: feature_cols}
            return self.feature_groups
    
    def select_representative_features(self, df, target):
        """
        Select representative features from each group based on correlation with target
        with proper handling of NaN values
        """
        self.selected_features = []
        
        correlations_with_target = []
        
        for group in self.feature_groups.values():
            # Calculate correlation with target for each feature in group
            correlations = []
            for feature in group:
                # Handle NaN values in correlation calculation
                correlation = abs(df[feature].fillna(df[feature].mean()).corr(df[target].fillna(df[target].mean())))
                if np.isfinite(correlation):  # Only add if correlation is finite
                    correlations.append((feature, correlation))
                    correlations_with_target.append((feature, correlation))
                else:
                    correlations.append((feature, 0))  # Use 0 correlation for invalid cases
            
            if correlations:  # Only process if we have valid correlations
                # Select feature with highest correlation with target
                best_feature = max(correlations, key=lambda x: x[1])[0]
                self.selected_features.append(best_feature)
        
        # Plot correlations with target
        self.plot_target_correlations(correlations_with_target)
        
        return self.selected_features
    
    def plot_correlation_matrix(self, df, feature_cols, figsize=(12, 8)):
        """
        Plot correlation matrix heatmap with proper handling of NaN values
        """
        print("Starting to plot correlation matrix")
        plt.figure(figsize=figsize)
        corr = self.calculate_correlations(df, feature_cols)
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        print("Finished plotting correlation matrix")
        return plt.gcf()
    
    def plot_feature_groups(self):
        """
        Plot feature groups as a bar chart
        """
        group_sizes = [len(group) for group in self.feature_groups.values()]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(group_sizes)), group_sizes)
        plt.title('Feature Group Sizes')
        plt.xlabel('Group ID')
        plt.ylabel('Number of Features')
        plt.show()
    
    def plot_target_correlations(self, correlations_with_target):
        """
        Plot correlations with target variable
        """
        features, correlations = zip(*correlations_with_target)
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(features)), correlations)
        plt.yticks(range(len(features)), features, fontsize=8)
        plt.title('Feature Correlations with Target')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.show()

class MarketPredictor:
    def __init__(self):
        self.feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.target = 'responder_6'
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols_final = None
        
    def create_temporal_features(self, df, symbol_id):
        """Creates time-based features with careful handling of NaN values"""
        windows = [5, 10, 20]
        temp_df = df.copy()
        feature_dfs = []
        
        # Get the symbol data once
        symbol_mask = temp_df['symbol_id'] == symbol_id
        symbol_data = temp_df[symbol_mask]
    
        for feat in self.feature_cols:
            feature_series = symbol_data[feat]
            
            for window in windows:
                # Calculate rolling statistics
                mean_series = feature_series.rolling(window, min_periods=1).mean()
                std_series = feature_series.rolling(window, min_periods=1).std()
                
                # Create a DataFrame with the new features
                window_df = pd.DataFrame({
                    f'{feat}_mean_{window}': mean_series,
                    f'{feat}_std_{window}': std_series
                }, index=symbol_data.index)
                
                feature_dfs.append(window_df)
        
        # Combine all new features
        if feature_dfs:
            all_features = pd.concat(feature_dfs, axis=1)
            
            # Update only the rows for the specified symbol_id
            temp_df.loc[symbol_mask, all_features.columns] = all_features
        
        return temp_df

    def create_lag_features(self, df, symbol_id, lags=[1, 2, 3]):
        """Creates lagged features with careful handling of NaN values"""
        temp_df = df.copy()
        feature_dfs = []
        
        # Get the symbol data once
        symbol_mask = temp_df['symbol_id'] == symbol_id
        symbol_data = temp_df[symbol_mask]
        
        for feat in self.feature_cols:
            feature_series = symbol_data[feat]
            lag_dict = {}
            
            for lag in lags:
                lag_dict[f'{feat}_lag_{lag}'] = feature_series.shift(lag)
            
            # Create a DataFrame with all lags for this feature
            lag_df = pd.DataFrame(lag_dict, index=symbol_data.index)
            feature_dfs.append(lag_df)
        
        # Combine all new features
        if feature_dfs:
            all_features = pd.concat(feature_dfs, axis=1)
            
            # Update only the rows for the specified symbol_id
            temp_df.loc[symbol_mask, all_features.columns] = all_features
        
        return temp_df
    def prepare_features(self, df, lags_df=None):
        """Prepares features with improved NaN handling"""
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
        """Gets all feature column names, including engineered features"""
        return [col for col in df.columns 
               if col.startswith('feature_') or 
                  col.endswith(('_mean_5', '_mean_10', '_mean_20',
                               '_std_5', '_std_10', '_std_20',
                               '_lag_1', '_lag_2', '_lag_3')) or
                  col.endswith('_prev')]

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from the model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            plt.figure(figsize=(12, 6))
            plt.barh(range(len(importance)), importance)
            plt.yticks(range(len(importance)), feature_names, fontsize=8)
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()

    def plot_predictions_vs_actual(self, y_true, y_pred):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.tight_layout()
        plt.show()

    def train(self, train_df, lags_df=None):
        """Modified train method with better error handling and reduced parallelism"""
        try:
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

            # Modified hyperparameter space
            param_distributions = {
                'n_estimators': randint(100, 1000),
                'learning_rate': uniform(0.01, 0.1),
                'max_depth': randint(3, 10),
                'num_leaves': randint(20, 50),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3),
                'min_child_samples': randint(1, 30),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }

            base_model = LGBMRegressor(
                random_state=42,
                verbose=-1
            )
            
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=30,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=1,
                verbose=1,
                random_state=42
            )

            X = df[self.feature_cols_final]
            y = df[self.target]

            if X.empty or y.empty:
                raise ValueError("Feature or target data is empty!")

            # Scale features before training
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            random_search.fit(X_scaled, y)
            best_model = random_search.best_estimator_
            
            # Plot feature importance
            self.plot_feature_importance(best_model, self.feature_cols_final)
            
            return best_model

        except Exception as e:
            print(f"Error in training: {str(e)}")
            # Fallback to a simpler model if the random search fails
            simple_model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            X = df[self.feature_cols_final]
            y = df[self.target]
            X_scaled = self.scaler.fit_transform(X)
            simple_model.fit(X_scaled, y)
            return simple_model

    def predict(self, test_df, lags_df=None):
        """Makes predictions on test data"""
        try:
            df = self.prepare_features(test_df, lags_df)
            if self.feature_cols_final is None:
                raise ValueError("Model has not been trained yet!")
            
            X = df[self.feature_cols_final]
            X_scaled = self.scaler.transform(X)
            predictions = self.models['ensemble'].predict(X_scaled)
            return predictions
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    def evaluate(self, y_true, y_pred, weights):
        """Calculates the weighted R-squared score"""
        weighted_mse = np.sum(weights * (y_true - y_pred) ** 2)
        weighted_var = np.sum(weights * y_true ** 2)
        r2 = 1 - weighted_mse / weighted_var
        return r2

def main():
    try:
        # File path
        path = "/kaggle/input/jane-street-real-time-market-data-forecasting"
        samples = []

        # Load data
        for i in range(1):
            file_path = f"{path}/train.parquet/partition_id={i}/part-0.parquet"
            try:
                part = pd.read_parquet(file_path)
                print(f"Loaded partition {i} with shape: {part.shape}")
                samples.append(part)
            except Exception as e:
                print(f"Error loading partition {i}: {str(e)}")

        # Transform into single dataframe
        df = pd.concat(samples, ignore_index=True)
        print(f"Combined DataFrame shape: {df.shape}")

        # Take a smaller subset for initial testing
        df = df.sample(n=min(len(df), 10000), random_state=42)
        print(f"Using subset of data with shape: {df.shape}")

        # Initialize Market Predictor
        predictor = MarketPredictor()
        
        # Split data into train and test sets
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Create lags dataframe for training
        responder_cols = ([f'responder_{i}' for i in range(6)] + 
                         [f'responder_{i}' for i in range(7, 9)])
        lags_df = train_df[responder_cols].shift(1)
        
        # Set up the plotting environment
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [12, 8]
        
        # Create a figure with subplots for data distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(train_df[predictor.target], bins=50)
        plt.title('Target Distribution - Training Data')
        plt.xlabel(predictor.target)
        
        plt.subplot(1, 2, 2)
        sns.histplot(test_df[predictor.target], bins=50)
        plt.title('Target Distribution - Test Data')
        plt.xlabel(predictor.target)
        plt.tight_layout()
        plt.show()
        
        # Train the model
        print("\nTraining model...")
        best_model = predictor.train(train_df, lags_df)
        predictor.models['ensemble'] = best_model
        
        # Prepare test data and make predictions
        print("\nMaking predictions...")
        test_lags_df = test_df[responder_cols].shift(1)
        test_prepared = predictor.prepare_features(test_df, test_lags_df)
        predictions = predictor.predict(test_df, test_lags_df)
        
        if predictions is not None:
            # Get the actual values, ensuring they align with predictions
            actuals = test_prepared[predictor.target]
            
            # Ensure predictions and actuals have the same length
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]
            
            # Calculate weights for evaluation
            weights = np.ones(min_len) / min_len
            
            # Model Evaluation with R^2
            r2_score = predictor.evaluate(
                actuals,
                predictions,
                weights
            )
            print(f"\nModel R-squared score: {r2_score}")
            print(f"Number of samples used in evaluation: {min_len}")
            
            # Plot predictions vs actual
            print("\nPlotting predictions vs actual values...")
            predictor.plot_predictions_vs_actual(actuals, predictions)
            
            # Plot prediction error distribution
            plt.figure(figsize=(12, 6))
            errors = predictions - actuals
            sns.histplot(errors, bins=50)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.show()
            
            # Plot rolling mean of absolute errors
            plt.figure(figsize=(12, 6))
            abs_errors = np.abs(errors)
            rolling_mae = pd.Series(abs_errors).rolling(window=100).mean()
            plt.plot(rolling_mae)
            plt.title('Rolling Mean Absolute Error (window=100)')
            plt.xlabel('Sample')
            plt.ylabel('Mean Absolute Error')
            plt.show()
            
            # Create a residual plot
            plt.figure(figsize=(12, 6))
            plt.scatter(predictions, errors, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.show()
            
        else:
            print("Failed to generate predictions")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise  # Re-raise the exception to see the full traceback
        
if __name__ == "__main__":
    main()
