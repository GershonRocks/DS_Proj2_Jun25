import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class LoanPredictionML:
    """
    Machine Learning Pipeline for Loan Prediction
    """

    def __init__(self, train_path, test_path):
        """
        Initialize the ML pipeline

        Parameters:
        train_path (str): Path to training CSV file
        test_path (str): Path to test CSV file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

        # Load data
        self.load_data()
        self.prepare_data()

    def load_data(self):
        """Load training and test datasets"""
        try:
            self.train_df = pd.read_csv(self.train_path)
            self.test_df = pd.read_csv(self.test_path)
            print(f"Training data loaded: {self.train_df.shape}")
            print(f"Test data loaded: {self.test_df.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def prepare_data(self):
        """Prepare features and target variables"""
        if self.train_df is None or self.test_df is None:
            return

        # Separate features and target
        self.X_train = self.train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        self.y_train = self.train_df['Loan_Status']
        self.X_test = self.test_df.drop(['Loan_ID'], axis=1)

        print("Data preparation completed!")
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Training target shape: {self.y_train.shape}")
        print(f"Test features shape: {self.X_test.shape}")

    def data_preprocessing(self):
        """Handle missing values and encode categorical variables"""
        print("=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)

        # Combine train and test for consistent preprocessing
        combined_df = pd.concat([self.X_train, self.X_test], axis=0, ignore_index=True)

        print("Missing values before preprocessing:")
        print(combined_df.isnull().sum())

        # Handle missing values
        # Numerical columns
        numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
        num_imputer = SimpleImputer(strategy='median')
        combined_df[numerical_cols] = num_imputer.fit_transform(combined_df[numerical_cols])

        # Categorical columns
        categorical_cols = combined_df.select_dtypes(include=['object']).columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        combined_df[categorical_cols] = cat_imputer.fit_transform(combined_df[categorical_cols])

        print("\nMissing values after preprocessing:")
        print(combined_df.isnull().sum())

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col])
            self.label_encoders[col] = le

        # Encode target variable
        target_le = LabelEncoder()
        self.y_train = target_le.fit_transform(self.y_train)
        self.label_encoders['Loan_Status'] = target_le

        # Split back to train and test
        train_size = len(self.X_train)
        self.X_train_processed = combined_df[:train_size]
        self.X_test_processed = combined_df[train_size:]

        # Scale features
        self.X_train_processed = self.scaler.fit_transform(self.X_train_processed)
        self.X_test_processed = self.scaler.transform(self.X_test_processed)

        print("Preprocessing completed!")
        print(f"Processed training features shape: {self.X_train_processed.shape}")
        print(f"Processed test features shape: {self.X_test_processed.shape}")

    def initialize_models(self):
        """Initialize different machine learning models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB()
        }
        print("Models initialized:")
        for name in self.models.keys():
            print(f"- {name}")

    def train_models(self):
        """Train all models and evaluate using cross-validation"""
        print("=" * 50)
        print("MODEL TRAINING & EVALUATION")
        print("=" * 50)

        if self.X_train_processed is None:
            self.data_preprocessing()

        self.initialize_models()

        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train_processed, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5, scoring='accuracy')

            # Train on full training set
            model.fit(self.X_train_processed, self.y_train)

            # Validation predictions
            val_pred = model.predict(X_val_split)
            val_accuracy = accuracy_score(y_val_split, val_pred)

            # Store scores
            self.model_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_accuracy': val_accuracy
            }

            print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Validation accuracy: {val_accuracy:.4f}")

    def model_comparison(self):
        """Compare model performances"""
        print("=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)

        comparison_df = pd.DataFrame(self.model_scores).T
        comparison_df = comparison_df.sort_values('cv_mean', ascending=False)

        print("Model Performance Comparison:")
        print(comparison_df.round(4).to_string())

        # Visualize model comparison
        plt.figure(figsize=(12, 6))

        # Cross-validation scores
        plt.subplot(1, 2, 1)
        models = list(comparison_df.index)
        cv_means = comparison_df['cv_mean']
        cv_stds = comparison_df['cv_std']

        plt.barh(models, cv_means, xerr=cv_stds, capsize=5)
        plt.xlabel('Cross-Validation Accuracy')
        plt.title('Model Performance (Cross-Validation)')
        plt.grid(True, alpha=0.3)

        # Validation scores
        plt.subplot(1, 2, 2)
        val_scores = comparison_df['val_accuracy']
        plt.barh(models, val_scores)
        plt.xlabel('Validation Accuracy')
        plt.title('Model Performance (Validation)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Select best model
        best_model_name = comparison_df.index[0]
        self.best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name}")
        print(f"Best CV Score: {comparison_df.loc[best_model_name, 'cv_mean']:.4f}")

        return best_model_name

    def detailed_evaluation(self, model_name=None):
        """Detailed evaluation of the best model"""
        if model_name is None:
            model_name = self.model_comparison()

        model = self.models[model_name]

        print("=" * 50)
        print(f"DETAILED EVALUATION: {model_name}")
        print("=" * 50)

        # Split for detailed evaluation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train_processed, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )

        # Predictions
        y_pred = model.predict(X_val_split)
        y_pred_proba = model.predict_proba(X_val_split)[:, 1]

        # Classification report
        print("Classification Report:")
        target_names = self.label_encoders['Loan_Status'].classes_
        print(classification_report(y_val_split, y_pred, target_names=target_names))

        # Confusion Matrix
        cm = confusion_matrix(y_val_split, y_pred)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_val_split, y_pred_proba)
        auc_score = roc_auc_score(y_val_split, y_pred_proba)

        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return auc_score

    def feature_importance(self, model_name=None):
        """Analyze feature importance for tree-based models"""
        if model_name is None:
            # Get best model name
            comparison_df = pd.DataFrame(self.model_scores).T
            model_name = comparison_df.sort_values('cv_mean', ascending=False).index[0]

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            print("=" * 50)
            print(f"FEATURE IMPORTANCE: {model_name}")
            print("=" * 50)

            # Get feature names
            feature_names = self.X_train.columns
            importances = model.feature_importances_

            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print("Feature Importance Ranking:")
            display(importance_df)

            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), y='Feature', x='Importance')
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")

    def hyperparameter_tuning(self, model_name='Random Forest', verbose=True):
        """Perform hyperparameter tuning for the specified model"""
        print("=" * 50)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("=" * 50)

        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)

        elif model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42)

        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return

        # Display parameter grid
        print("Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal parameter combinations to test: {total_combinations}")
        print(f"With 5-fold CV, total model fits: {total_combinations * 5}")
        print("\nStarting Grid Search...")
        print("-" * 30)

        # Grid search with maximum verbosity
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=3, return_train_score=True
        )

        import time
        start_time = time.time()
        grid_search.fit(self.X_train_processed, self.y_train)
        end_time = time.time()

        print(f"\nGrid Search completed in {end_time - start_time:.2f} seconds")
        print("=" * 50)
        print("HYPERPARAMETER TUNING RESULTS")
        print("=" * 50)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Best validation std: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")

        # Show top 10 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_results = results_df.nlargest(10, 'mean_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]

        print("\nTop 10 Parameter Combinations:")
        print(top_results.to_string(index=False))

        # Visualize results if possible
        if len(param_grid) <= 2:
            self._plot_hyperparameter_results(grid_search, param_grid, model_name)

        # Compare with original model
        if model_name in self.models:
            original_model = self.models[model_name]
            # Get original model CV score
            original_cv_scores = cross_val_score(original_model, self.X_train_processed, self.y_train, cv=5)
            original_mean = original_cv_scores.mean()

            print(f"\nPerformance Comparison:")
            print(f"Original {model_name}: {original_mean:.4f}")
            print(f"Tuned {model_name}: {grid_search.best_score_:.4f}")
            print(f"Improvement: {grid_search.best_score_ - original_mean:.4f}")

        # Update model with best parameters
        self.models[f'{model_name}_Tuned'] = grid_search.best_estimator_

        # Update model scores
        self.model_scores[f'{model_name}_Tuned'] = {
            'cv_mean': grid_search.best_score_,
            'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'val_accuracy': grid_search.best_score_  # Placeholder
        }

        return grid_search.best_estimator_

    def _plot_hyperparameter_results(self, grid_search, param_grid, model_name):
        """Plot hyperparameter tuning results for visualization"""
        results_df = pd.DataFrame(grid_search.cv_results_)

        if len(param_grid) == 1:
            # Single parameter
            param_name = list(param_grid.keys())[0]
            param_values = [p[param_name] for p in results_df['params']]

            plt.figure(figsize=(10, 6))
            plt.plot(param_values, results_df['mean_test_score'], 'bo-')
            plt.fill_between(param_values,
                             results_df['mean_test_score'] - results_df['std_test_score'],
                             results_df['mean_test_score'] + results_df['std_test_score'],
                             alpha=0.2)
            plt.xlabel(param_name)
            plt.ylabel('Cross-Validation Score')
            plt.title(f'Hyperparameter Tuning Results - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.show()

        elif len(param_grid) == 2:
            # Two parameters - heatmap
            param_names = list(param_grid.keys())
            param1_name, param2_name = param_names[0], param_names[1]

            # Create pivot table
            pivot_data = []
            for i, params in enumerate(results_df['params']):
                pivot_data.append({
                    param1_name: params[param1_name],
                    param2_name: params[param2_name],
                    'score': results_df['mean_test_score'].iloc[i]
                })

            pivot_df = pd.DataFrame(pivot_data)
            pivot_table = pivot_df.pivot(index=param2_name, columns=param1_name, values='score')

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
            plt.title(f'Hyperparameter Tuning Heatmap - {model_name}')
            plt.show()

    def generate_predictions(self, model_name=None):
        """Generate predictions for test dataset"""
        if model_name is None:
            # Use best model
            comparison_df = pd.DataFrame(self.model_scores).T
            model_name = comparison_df.sort_values('cv_mean', ascending=False).index[0]

        model = self.models[model_name]

        print("=" * 50)
        print(f"GENERATING PREDICTIONS: {model_name}")
        print("=" * 50)

        # Make predictions
        predictions = model.predict(self.X_test_processed)
        prediction_proba = model.predict_proba(self.X_test_processed)

        # Convert back to original labels
        predictions_original = self.label_encoders['Loan_Status'].inverse_transform(predictions)

        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Loan_ID': self.test_df['Loan_ID'],
            'Loan_Status': predictions_original,
            'Probability_Approved': prediction_proba[:, 1]
        })

        print("Prediction Summary:")
        print(submission_df['Loan_Status'].value_counts())
        print(f"\nApproval Rate: {(submission_df['Loan_Status'] == 'Y').mean():.2%}")

        # Save predictions
        submission_df.to_csv('loan_predictions.csv', index=False)
        print("\nPredictions saved to 'loan_predictions.csv'")

        return submission_df

    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("LOAN PREDICTION ML PIPELINE")
        print("=" * 80)

        # Preprocessing
        self.data_preprocessing()

        # Train models
        self.train_models()

        # Compare models
        best_model_name = self.model_comparison()

        # Detailed evaluation
        self.detailed_evaluation(best_model_name)

        # Feature importance
        self.feature_importance(best_model_name)

        # Generate predictions
        predictions = self.generate_predictions(best_model_name)

        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return predictions

# Usage Example:
# ml_pipeline = LoanPredictionML(
#     train_path='datasets/altruistdelhite04/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv',
#     test_path='datasets/altruistdelhite04/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv'
# )
#
# # Run complete pipeline
# predictions = ml_pipeline.run_complete_pipeline()
#
# # Or run individual steps:
# ml_pipeline.data_preprocessing()
# ml_pipeline.train_models()
# ml_pipeline.model_comparison()
# ml_pipeline.detailed_evaluation()
# ml_pipeline.hyperparameter_tuning('Random Forest')
# predictions = ml_pipeline.generate_predictions()
