import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class LoanDatasetEDA:
    """
    Exploratory Data Analysis class for Loan Prediction Dataset
    """

    def __init__(self, file_path):
        """
        Initialize the EDA class with the dataset

        Parameters:
        file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = 'Loan_Status'

        # Load and prepare the dataset
        self.load_data()
        self.identify_column_types()

    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def identify_column_types(self):
        """Identify numerical and categorical columns"""
        if self.df is not None:
            self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

            # Remove target column from categorical if present
            if self.target_col in self.categorical_cols:
                self.categorical_cols.remove(self.target_col)

    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            return

        print("=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)

        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1]}")
        print(f"Number of Records: {self.df.shape[0]}")

        print("\nColumn Information:")
        print(self.df.info())

        print(f"\nNumerical Columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"Categorical Columns ({len(self.categorical_cols)}): {self.categorical_cols}")

        print("\nFirst 5 rows:")
        display(self.df.head())

        print("\nLast 5 rows:")
        display(self.df.tail())

    def missing_values_analysis(self):
        """Analyze missing values in the dataset"""
        if self.df is None:
            return

        print("=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)

        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        })

        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

        if len(missing_df) > 0:
            print("Columns with missing values:")
            display(missing_df)

            # Visualize missing values
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.show()
        else:
            print("No missing values found in the dataset!")

    def numerical_analysis(self):
        """Analyze numerical columns"""
        if self.df is None or len(self.numerical_cols) == 0:
            return

        print("=" * 50)
        print("NUMERICAL FEATURES ANALYSIS")
        print("=" * 50)

        print("Descriptive Statistics:")
        display(self.df[self.numerical_cols].describe().round(2))

        # Distribution plots
        n_cols = len(self.numerical_cols)
        if n_cols > 0:
            fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 5 * ((n_cols + 1) // 2)))
            if n_cols == 1:
                axes = [axes]
            elif (n_cols + 1) // 2 == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, col in enumerate(self.numerical_cols):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)

            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

        # Box plots for outlier detection
        if len(self.numerical_cols) > 0:
            fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 5 * ((n_cols + 1) // 2)))
            if n_cols == 1:
                axes = [axes]
            elif (n_cols + 1) // 2 == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, col in enumerate(self.numerical_cols):
                sns.boxplot(data=self.df, y=col, ax=axes[i])
                axes[i].set_title(f'Box Plot of {col}')
                axes[i].grid(True, alpha=0.3)

            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

    def categorical_analysis(self):
        """Analyze categorical columns"""
        if self.df is None or len(self.categorical_cols) == 0:
            return

        print("=" * 50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("=" * 50)

        for col in self.categorical_cols:
            print(f"\n{col} - Value Counts:")
            value_counts = self.df[col].value_counts(dropna=False)
            value_percent = self.df[col].value_counts(normalize=True, dropna=False) * 100

            summary_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percent.round(2)
            })
            display(summary_df)

        # Visualize categorical distributions
        n_cols = len(self.categorical_cols)
        if n_cols > 0:
            fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 5 * ((n_cols + 1) // 2)))
            if n_cols == 1:
                axes = [axes]
            elif (n_cols + 1) // 2 == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, col in enumerate(self.categorical_cols):
                self.df[col].value_counts().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

    def target_analysis(self):
        """Analyze target variable"""
        if self.df is None or self.target_col not in self.df.columns:
            print(f"Target column '{self.target_col}' not found in dataset")
            return

        print("=" * 50)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 50)

        print(f"Target Variable: {self.target_col}")
        target_counts = self.df[self.target_col].value_counts()
        target_percent = self.df[self.target_col].value_counts(normalize=True) * 100

        target_summary = pd.DataFrame({
            'Count': target_counts,
            'Percentage': target_percent.round(2)
        })
        display(target_summary)

        # Visualize target distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot
        target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title(f'Distribution of {self.target_col}')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(True, alpha=0.3)

        # Pie chart
        ax2.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=['skyblue', 'lightcoral'])
        ax2.set_title(f'{self.target_col} Distribution')

        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        if self.df is None or len(self.numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return

        print("=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        # Calculate correlation matrix
        corr_matrix = self.df[self.numerical_cols].corr()

        print("Correlation Matrix:")
        display(corr_matrix.round(3))

        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j].round(3)
                    })

        if high_corr_pairs:
            print("\nHighly Correlated Feature Pairs (|correlation| > 0.5):")
            display(pd.DataFrame(high_corr_pairs))
        else:
            print("\nNo highly correlated feature pairs found (threshold: 0.5)")

    def bivariate_analysis(self):
        """Analyze relationships between features and target variable"""
        if self.df is None or self.target_col not in self.df.columns:
            return

        print("=" * 50)
        print("BIVARIATE ANALYSIS WITH TARGET")
        print("=" * 50)

        # Numerical features vs target
        if self.numerical_cols:
            print("Numerical Features vs Target:")
            for col in self.numerical_cols:
                print(f"\n{col} by {self.target_col}:")
                grouped_stats = self.df.groupby(self.target_col)[col].agg(['count', 'mean', 'median', 'std']).round(2)
                display(grouped_stats)

            # Visualize numerical features vs target
            n_num_cols = len(self.numerical_cols)
            if n_num_cols > 0:
                fig, axes = plt.subplots(nrows=(n_num_cols + 1) // 2, ncols=2,
                                         figsize=(15, 5 * ((n_num_cols + 1) // 2)))
                if n_num_cols == 1:
                    axes = [axes]
                elif (n_num_cols + 1) // 2 == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, col in enumerate(self.numerical_cols):
                    sns.boxplot(data=self.df, x=self.target_col, y=col, ax=axes[i])
                    axes[i].set_title(f'{col} by {self.target_col}')
                    axes[i].grid(True, alpha=0.3)

                # Hide empty subplots
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                plt.tight_layout()
                plt.show()

        # Categorical features vs target
        if self.categorical_cols:
            print("\nCategorical Features vs Target:")
            for col in self.categorical_cols:
                print(f"\n{col} vs {self.target_col} - Cross Tabulation:")
                cross_tab = pd.crosstab(self.df[col], self.df[self.target_col], margins=True)
                display(cross_tab)

                # Calculate percentages
                cross_tab_pct = pd.crosstab(self.df[col], self.df[self.target_col], normalize='index') * 100
                print(f"\n{col} vs {self.target_col} - Percentage Distribution:")
                display(cross_tab_pct.round(2))

            # Visualize categorical features vs target
            n_cat_cols = len(self.categorical_cols)
            if n_cat_cols > 0:
                fig, axes = plt.subplots(nrows=(n_cat_cols + 1) // 2, ncols=2,
                                         figsize=(15, 5 * ((n_cat_cols + 1) // 2)))
                if n_cat_cols == 1:
                    axes = [axes]
                elif (n_cat_cols + 1) // 2 == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, col in enumerate(self.categorical_cols):
                    cross_tab_pct = pd.crosstab(self.df[col], self.df[self.target_col], normalize='index') * 100
                    cross_tab_pct.plot(kind='bar', ax=axes[i], stacked=True)
                    axes[i].set_title(f'{col} vs {self.target_col}')
                    axes[i].set_ylabel('Percentage')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title=self.target_col)
                    axes[i].grid(True, alpha=0.3)

                # Hide empty subplots
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                plt.tight_layout()
                plt.show()

    def outlier_analysis(self):
        """Detect and analyze outliers in numerical features"""
        if self.df is None or len(self.numerical_cols) == 0:
            return

        print("=" * 50)
        print("OUTLIER ANALYSIS")
        print("=" * 50)

        outlier_summary = []

        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.df)) * 100

            outlier_summary.append({
                'Feature': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': round(outlier_percentage, 2),
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2)
            })

        outlier_df = pd.DataFrame(outlier_summary)
        display(outlier_df)

    def generate_full_report(self):
        """Generate complete EDA report"""
        print("COMPREHENSIVE EDA REPORT")
        print("=" * 80)

        self.basic_info()
        self.missing_values_analysis()
        self.numerical_analysis()
        self.categorical_analysis()
        self.target_analysis()
        self.correlation_analysis()
        self.bivariate_analysis()
        self.outlier_analysis()

        print("=" * 80)
        print("EDA REPORT COMPLETED")
        print("=" * 80)

# Usage Example:
# eda = LoanDatasetEDA('datasets/altruistdelhite04/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
# eda.generate_full_report()

# Individual analysis methods can also be called:
# eda.basic_info()
# eda.missing_values_analysis()
# eda.target_analysis()
# etc.