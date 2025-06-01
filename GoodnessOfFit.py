import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings('ignore')


class GoodnessOfFit:
    """
    Comprehensive Goodness of Fit evaluation class for classification models
    """

    def __init__(self, ml_pipeline):
        """
        Initialize with the ML pipeline object

        Parameters:
        ml_pipeline: LoanPredictionML object with trained models
        """
        self.ml_pipeline = ml_pipeline
        self.results = {}

    def evaluate_model(self, model_name, X_test=None, y_test=None, test_size=0.2):
        """
        Comprehensive evaluation of a specific model

        Parameters:
        model_name (str): Name of the model to evaluate
        X_test, y_test: Optional test set (if None, will split from training data)
        test_size (float): Size of test split if creating from training data
        """

        if model_name not in self.ml_pipeline.models:
            print(f"Model '{model_name}' not found in pipeline")
            return

        model = self.ml_pipeline.models[model_name]

        # Prepare test data
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.ml_pipeline.X_train_processed,
                self.ml_pipeline.y_train,
                test_size=test_size,
                random_state=42,
                stratify=self.ml_pipeline.y_train
            )

        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate all metrics
        results = self._calculate_all_metrics(y_test, y_pred, y_pred_proba)
        results['model_name'] = model_name

        # Store results
        self.results[model_name] = results

        return results

    def _calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive set of evaluation metrics"""

        metrics = {}

        # Basic Classification Metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

        # Confusion Matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negative'] = tn
            metrics['false_positive'] = fp
            metrics['false_negative'] = fn
            metrics['true_positive'] = tp
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['precision_positive'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall_positive'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # AUC and ROC
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)

        # Matthews Correlation Coefficient
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Probabilistic metrics
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)

        # Lift calculation (for top decile)
        lift_score = self._calculate_lift(y_true, y_pred_proba)
        metrics['lift_top_decile'] = lift_score

        return metrics

    def _calculate_lift(self, y_true, y_pred_proba, decile=10):
        """Calculate lift for top decile"""
        df_temp = pd.DataFrame({
            'actual': y_true,
            'predicted_prob': y_pred_proba
        })

        # Sort by predicted probability
        df_temp = df_temp.sort_values('predicted_prob', ascending=False)

        # Calculate decile
        decile_size = len(df_temp) // decile
        top_decile = df_temp.head(decile_size)

        # Calculate lift
        baseline_rate = y_true.mean()
        top_decile_rate = top_decile['actual'].mean()

        lift = top_decile_rate / baseline_rate if baseline_rate > 0 else 0
        return lift

    def display_metrics_summary(self, model_name):
        """Display comprehensive metrics summary for a model"""

        if model_name not in self.results:
            print(f"No results found for {model_name}. Run evaluate_model() first.")
            return

        results = self.results[model_name]

        print("=" * 60)
        print(f"GOODNESS OF FIT EVALUATION: {model_name}")
        print("=" * 60)

        # Core Classification Metrics
        print("\n📊 CORE CLASSIFICATION METRICS")
        print("-" * 40)
        print(f"Accuracy:           {results['accuracy']:.4f}")
        print(f"Precision:          {results['precision']:.4f}")
        print(f"Recall:             {results['recall']:.4f}")
        print(f"F1-Score:           {results['f1_score']:.4f}")

        # Confusion Matrix Metrics
        if 'true_positive' in results:
            print(f"\n📈 CONFUSION MATRIX METRICS")
            print("-" * 40)
            print(f"True Positives:     {results['true_positive']}")
            print(f"True Negatives:     {results['true_negative']}")
            print(f"False Positives:    {results['false_positive']}")
            print(f"False Negatives:    {results['false_negative']}")
            print(f"Sensitivity (TPR):  {results['sensitivity']:.4f}")
            print(f"Specificity (TNR):  {results['specificity']:.4f}")

        # Advanced Metrics
        print(f"\n🎯 ADVANCED METRICS")
        print("-" * 40)
        print(f"ROC AUC:            {results['roc_auc']:.4f}")
        print(f"Average Precision:  {results['average_precision']:.4f}")
        print(f"Matthews Coeff:     {results['matthews_corrcoef']:.4f}")
        print(f"Cohen's Kappa:      {results['cohen_kappa']:.4f}")

        # Probabilistic Metrics
        print(f"\n📊 PROBABILISTIC METRICS")
        print("-" * 40)
        print(f"Log Loss:           {results['log_loss']:.4f}")
        print(f"Brier Score:        {results['brier_score']:.4f}")

        # Business Metrics
        print(f"\n💼 BUSINESS METRICS")
        print("-" * 40)
        print(f"Lift (Top 10%):     {results['lift_top_decile']:.4f}")

    def plot_comprehensive_evaluation(self, model_name, X_test=None, y_test=None):
        """Create comprehensive visualization plots"""

        if model_name not in self.ml_pipeline.models:
            print(f"Model '{model_name}' not found")
            return

        model = self.ml_pipeline.models[model_name]

        # Prepare test data if not provided
        if X_test is None or y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.ml_pipeline.X_train_processed,
                self.ml_pipeline.y_train,
                test_size=0.2,
                random_state=42,
                stratify=self.ml_pipeline.y_train
            )

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Model Evaluation: {model_name}', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        axes[0, 2].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Rejected', density=True)
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Approved', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{model_name}')
        axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Lift Chart
        self._plot_lift_chart(y_test, y_pred_proba, axes[1, 2])

        plt.tight_layout()
        plt.show()

    def _plot_lift_chart(self, y_true, y_pred_proba, ax):
        """Plot lift chart"""
        # Create dataframe and sort by probability
        df_lift = pd.DataFrame({
            'actual': y_true,
            'predicted_prob': y_pred_proba
        }).sort_values('predicted_prob', ascending=False)

        # Calculate cumulative lift
        deciles = np.arange(0.1, 1.1, 0.1)
        lift_values = []

        baseline_rate = y_true.mean()

        for decile in deciles:
            n_samples = int(len(df_lift) * decile)
            top_samples = df_lift.head(n_samples)
            rate = top_samples['actual'].mean()
            lift = rate / baseline_rate if baseline_rate > 0 else 0
            lift_values.append(lift)

        ax.plot(deciles * 100, lift_values, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax.set_xlabel('Percentage of Population')
        ax.set_ylabel('Lift')
        ax.set_title('Lift Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def compare_models(self, model_names=None):
        """Compare multiple models across all metrics"""

        if model_names is None:
            model_names = list(self.ml_pipeline.models.keys())

        # Evaluate all models if not already done
        for model_name in model_names:
            if model_name not in self.results:
                self.evaluate_model(model_name)

        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            if model_name in self.results:
                row = self.results[model_name].copy()
                row['Model'] = model_name
                comparison_data.append(row)

        if not comparison_data:
            print("No model results found. Run evaluate_model() first.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # Select key metrics for comparison
        key_metrics = ['Model', 'accuracy', 'precision', 'recall', 'f1_score',
                       'roc_auc', 'matthews_corrcoef', 'lift_top_decile']

        comparison_summary = comparison_df[key_metrics].round(4)
        comparison_summary = comparison_summary.sort_values('roc_auc', ascending=False)

        print("=" * 80)
        print("MODEL COMPARISON - GOODNESS OF FIT")
        print("=" * 80)
        print(comparison_summary.to_string(index=False))

        # Plot comparison
        self._plot_model_comparison(comparison_summary)

        return comparison_summary

    def _plot_model_comparison(self, comparison_df):
        """Plot model comparison across key metrics"""

        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'matthews_corrcoef']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def comprehensive_error_analysis(self, model_name, verbose=True):
        """
        Comprehensive error analysis answering key questions:
        - Do we have weak segments?
        - Can we predict the error?
        - Any patterns?
        - Special values?
        """

        if model_name not in self.ml_pipeline.models:
            print(f"Model '{model_name}' not found")
            return

        model = self.ml_pipeline.models[model_name]

        # Prepare test data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.ml_pipeline.X_train_processed,
            self.ml_pipeline.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.ml_pipeline.y_train
        )

        # Get original feature data for analysis
        X_train_orig, X_test_orig, _, _ = train_test_split(
            self.ml_pipeline.X_train,
            self.ml_pipeline.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.ml_pipeline.y_train
        )

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create error flags
        errors = (y_pred != y_test)
        correct_predictions = ~errors

        print("=" * 80)
        print(f"COMPREHENSIVE ERROR ANALYSIS: {model_name}")
        print("=" * 80)

        # Overall Error Statistics
        total_errors = np.sum(errors)
        total_predictions = len(y_test)
        error_rate = total_errors / total_predictions

        print(f"\n📊 OVERALL ERROR STATISTICS")
        print("-" * 50)
        print(f"Total Predictions:        {total_predictions}")
        print(f"Total Errors:             {total_errors}")
        print(f"Overall Error Rate:       {error_rate:.4f} ({error_rate * 100:.2f}%)")
        print(f"Accuracy:                 {1 - error_rate:.4f} ({(1 - error_rate) * 100:.2f}%)")

        # Error Type Analysis
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        false_negatives = np.sum((y_pred == 0) & (y_test == 1))

        print(f"\n❌ ERROR TYPE BREAKDOWN")
        print("-" * 50)
        print(f"False Positives (Type I):  {false_positives} ({false_positives / total_predictions * 100:.2f}%)")
        print(f"  → Incorrectly approved loans that should be rejected")
        print(f"False Negatives (Type II): {false_negatives} ({false_negatives / total_predictions * 100:.2f}%)")
        print(f"  → Incorrectly rejected loans that should be approved")

        if verbose:
            print(f"\n💡 BUSINESS IMPACT INTERPRETATION:")
            print(f"  • False Positives = Potential loan defaults ($$ RISK)")
            print(f"  • False Negatives = Lost business opportunities ($ LOSS)")
            if false_positives > false_negatives:
                print(f"  ⚠️  Model is more risky - approving bad loans!")
            elif false_negatives > false_positives:
                print(f"  ⚠️  Model is too conservative - rejecting good loans!")
            else:
                print(f"  ✅ Model has balanced error types")

        # 1. WEAK SEGMENTS ANALYSIS
        print(f"\n🎯 1. WEAK SEGMENTS ANALYSIS")
        print("=" * 50)

        segment_analysis = self._analyze_weak_segments(X_test_orig, y_test, y_pred, errors, verbose)

        # 2. ERROR PREDICTABILITY ANALYSIS
        print(f"\n🔮 2. CAN WE PREDICT THE ERROR?")
        print("=" * 50)

        error_predictability = self._analyze_error_predictability(X_test, y_test, y_pred, y_pred_proba, errors, verbose)

        # 3. ERROR PATTERNS ANALYSIS
        print(f"\n🔍 3. ERROR PATTERNS ANALYSIS")
        print("=" * 50)

        pattern_analysis = self._analyze_error_patterns(X_test_orig, y_test, y_pred, y_pred_proba, errors, verbose)

        # 4. SPECIAL VALUES ANALYSIS
        print(f"\n⭐ 4. SPECIAL VALUES ANALYSIS")
        print("=" * 50)

        special_values_analysis = self._analyze_special_values(X_test_orig, y_test, y_pred, errors, verbose)

        # 5. CONFIDENCE-ERROR RELATIONSHIP
        print(f"\n📈 5. CONFIDENCE vs ERROR RELATIONSHIP")
        print("=" * 50)

        confidence_analysis = self._analyze_confidence_errors(y_pred_proba, errors, verbose)

        # 6. FEATURE-SPECIFIC ERROR ANALYSIS
        print(f"\n🔧 6. FEATURE-SPECIFIC ERROR ANALYSIS")
        print("=" * 50)

        feature_error_analysis = self._analyze_feature_errors(X_test_orig, errors, verbose)

        # Summary and Recommendations
        print(f"\n🎯 SUMMARY & RECOMMENDATIONS")
        print("=" * 50)
        self._provide_error_recommendations(segment_analysis, error_predictability,
                                            pattern_analysis, special_values_analysis, verbose)

        return {
            'segment_analysis': segment_analysis,
            'error_predictability': error_predictability,
            'pattern_analysis': pattern_analysis,
            'special_values_analysis': special_values_analysis,
            'confidence_analysis': confidence_analysis,
            'feature_error_analysis': feature_error_analysis
        }

    def _analyze_weak_segments(self, X_test_orig, y_test, y_pred, errors, verbose=True):
        """Analyze which segments have high error rates"""

        segment_results = {}

        # Analyze categorical variables
        categorical_cols = X_test_orig.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if verbose:
                print(f"\n📊 Segment Analysis: {col}")
                print("-" * 30)

            segment_errors = {}
            for value in X_test_orig[col].unique():
                mask = X_test_orig[col] == value
                segment_error_rate = np.mean(errors[mask])
                segment_size = np.sum(mask)
                segment_errors[value] = {
                    'error_rate': segment_error_rate,
                    'size': segment_size,
                    'total_errors': np.sum(errors[mask])
                }

                if verbose:
                    print(
                        f"  {value:15} | Error Rate: {segment_error_rate:.3f} | Size: {segment_size:3d} | Errors: {np.sum(errors[mask]):2d}")

            # Find weakest segment
            weakest_segment = max(segment_errors.keys(), key=lambda x: segment_errors[x]['error_rate'])
            strongest_segment = min(segment_errors.keys(), key=lambda x: segment_errors[x]['error_rate'])

            if verbose:
                print(
                    f"  🔴 WEAKEST:  {weakest_segment} ({segment_errors[weakest_segment]['error_rate']:.3f} error rate)")
                print(
                    f"  🟢 STRONGEST: {strongest_segment} ({segment_errors[strongest_segment]['error_rate']:.3f} error rate)")

            segment_results[col] = segment_errors

        # Analyze numerical variables (quartiles)
        numerical_cols = X_test_orig.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if verbose:
                print(f"\n📊 Segment Analysis: {col} (Quartiles)")
                print("-" * 30)

            quartiles = np.percentile(X_test_orig[col], [25, 50, 75])
            quartile_labels = ['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)']
            quartile_errors = {}

            for i, label in enumerate(quartile_labels):
                if i == 0:
                    mask = X_test_orig[col] <= quartiles[0]
                elif i == 1:
                    mask = (X_test_orig[col] > quartiles[0]) & (X_test_orig[col] <= quartiles[1])
                elif i == 2:
                    mask = (X_test_orig[col] > quartiles[1]) & (X_test_orig[col] <= quartiles[2])
                else:
                    mask = X_test_orig[col] > quartiles[2]

                segment_error_rate = np.mean(errors[mask])
                segment_size = np.sum(mask)
                quartile_errors[label] = {
                    'error_rate': segment_error_rate,
                    'size': segment_size,
                    'total_errors': np.sum(errors[mask])
                }

                if verbose:
                    print(
                        f"  {label:12} | Error Rate: {segment_error_rate:.3f} | Size: {segment_size:3d} | Errors: {np.sum(errors[mask]):2d}")

            segment_results[f"{col}_quartiles"] = quartile_errors

        return segment_results

    def _analyze_error_predictability(self, X_test, y_test, y_pred, y_pred_proba, errors, verbose=True):
        """Analyze if we can predict where errors will occur"""

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Train a model to predict errors
        try:
            error_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            error_predictor.fit(X_test, errors)

            error_pred = error_predictor.predict(X_test)
            error_pred_proba = error_predictor.predict_proba(X_test)[:, 1]

            error_prediction_accuracy = accuracy_score(errors, error_pred)

            if len(np.unique(errors)) > 1:  # Check if we have both classes
                error_prediction_auc = roc_auc_score(errors, error_pred_proba)
            else:
                error_prediction_auc = 0.5

            if verbose:
                print(f"Error Prediction Accuracy:    {error_prediction_accuracy:.4f}")
                print(f"Error Prediction AUC:         {error_prediction_auc:.4f}")

                if error_prediction_auc > 0.7:
                    print("✅ ERRORS ARE HIGHLY PREDICTABLE!")
                    print("   → We can identify high-risk predictions")
                    print("   → Consider using prediction confidence thresholds")
                elif error_prediction_auc > 0.6:
                    print("⚠️  ERRORS ARE SOMEWHAT PREDICTABLE")
                    print("   → Some patterns exist but not very strong")
                else:
                    print("❌ ERRORS APPEAR RANDOM")
                    print("   → No clear pattern in where errors occur")

            # Feature importance for error prediction
            feature_importance = error_predictor.feature_importances_
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]

            if hasattr(self.ml_pipeline, 'X_train') and hasattr(self.ml_pipeline.X_train, 'columns'):
                feature_names = self.ml_pipeline.X_train.columns

            top_error_features = sorted(zip(feature_names, feature_importance),
                                        key=lambda x: x[1], reverse=True)[:5]

            if verbose:
                print(f"\nTop 5 features that predict errors:")
                for i, (feature, importance) in enumerate(top_error_features, 1):
                    print(f"  {i}. {feature}: {importance:.4f}")

            return {
                'predictability_accuracy': error_prediction_accuracy,
                'predictability_auc': error_prediction_auc,
                'top_error_features': top_error_features,
                'is_predictable': error_prediction_auc > 0.6
            }

        except Exception as e:
            if verbose:
                print(f"Could not analyze error predictability: {e}")
            return {'predictability_accuracy': 0, 'predictability_auc': 0.5, 'is_predictable': False}

    def _analyze_error_patterns(self, X_test_orig, y_test, y_pred, y_pred_proba, errors, verbose=True):
        """Analyze patterns in errors"""

        patterns = {}

        # Confidence patterns
        error_confidence = y_pred_proba[errors]
        correct_confidence = y_pred_proba[~errors]

        patterns['confidence'] = {
            'error_mean_confidence': np.mean(error_confidence),
            'correct_mean_confidence': np.mean(correct_confidence),
            'error_std_confidence': np.std(error_confidence),
            'correct_std_confidence': np.std(correct_confidence)
        }

        if verbose:
            print(f"Average confidence for ERRORS:     {np.mean(error_confidence):.4f}")
            print(f"Average confidence for CORRECT:    {np.mean(correct_confidence):.4f}")
            print(f"Confidence std for ERRORS:         {np.std(error_confidence):.4f}")
            print(f"Confidence std for CORRECT:        {np.std(correct_confidence):.4f}")

            if np.mean(error_confidence) < 0.6:
                print("✅ Good! Errors tend to have lower confidence")
                print("   → Model is appropriately uncertain about wrong predictions")
            else:
                print("⚠️  Concerning! Errors have high confidence")
                print("   → Model is overconfident in wrong predictions")

        # Class-specific error patterns
        false_positive_mask = (y_pred == 1) & (y_test == 0)
        false_negative_mask = (y_pred == 0) & (y_test == 1)

        if np.sum(false_positive_mask) > 0:
            fp_confidence = y_pred_proba[false_positive_mask]
            patterns['false_positive_confidence'] = {
                'mean': np.mean(fp_confidence),
                'std': np.std(fp_confidence)
            }

            if verbose:
                print(f"\nFalse Positive Analysis:")
                print(f"  Average confidence: {np.mean(fp_confidence):.4f}")
                print(f"  → Model confidence when incorrectly approving loans")

        if np.sum(false_negative_mask) > 0:
            fn_confidence = y_pred_proba[false_negative_mask]
            patterns['false_negative_confidence'] = {
                'mean': np.mean(fn_confidence),
                'std': np.std(fn_confidence)
            }

            if verbose:
                print(f"\nFalse Negative Analysis:")
                print(f"  Average confidence: {np.mean(fn_confidence):.4f}")
                print(f"  → Model confidence when incorrectly rejecting loans")

        return patterns

    def _analyze_special_values(self, X_test_orig, y_test, y_pred, errors, verbose=True):
        """Analyze errors related to special values"""

        special_analysis = {}

        # Check for missing value patterns (after imputation, look for imputed values)
        if verbose:
            print("Looking for special value patterns...")

        numerical_cols = X_test_orig.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            col_data = X_test_orig[col]

            # Check for values at quartiles (potential imputed values)
            median_val = np.median(col_data)
            mean_val = np.mean(col_data)

            # Values exactly at median (possible imputation)
            median_mask = np.abs(col_data - median_val) < 1e-10
            if np.sum(median_mask) > len(col_data) * 0.05:  # More than 5% at exact median
                median_error_rate = np.mean(errors[median_mask])
                other_error_rate = np.mean(errors[~median_mask])

                special_analysis[f"{col}_median_values"] = {
                    'count': np.sum(median_mask),
                    'error_rate': median_error_rate,
                    'other_error_rate': other_error_rate
                }

                if verbose:
                    print(f"\n🔍 {col} - Potential imputed values at median:")
                    print(f"  Count at median: {np.sum(median_mask)}")
                    print(f"  Error rate at median: {median_error_rate:.4f}")
                    print(f"  Error rate elsewhere: {other_error_rate:.4f}")

            # Check for outliers
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            outlier_mask = (col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)

            if np.sum(outlier_mask) > 0:
                outlier_error_rate = np.mean(errors[outlier_mask])
                normal_error_rate = np.mean(errors[~outlier_mask])

                special_analysis[f"{col}_outliers"] = {
                    'count': np.sum(outlier_mask),
                    'error_rate': outlier_error_rate,
                    'normal_error_rate': normal_error_rate
                }

                if verbose:
                    print(f"\n📊 {col} - Outlier analysis:")
                    print(f"  Outlier count: {np.sum(outlier_mask)}")
                    print(f"  Error rate for outliers: {outlier_error_rate:.4f}")
                    print(f"  Error rate for normal values: {normal_error_rate:.4f}")

                    if outlier_error_rate > normal_error_rate * 1.5:
                        print(f"  ⚠️  OUTLIERS HAVE MUCH HIGHER ERROR RATE!")
                    elif outlier_error_rate < normal_error_rate * 0.5:
                        print(f"  ✅ Outliers actually perform better")

        return special_analysis

    def _analyze_confidence_errors(self, y_pred_proba, errors, verbose=True):
        """Analyze relationship between prediction confidence and errors"""

        # Bin by confidence levels
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_labels = [f"{confidence_bins[i]:.1f}-{confidence_bins[i + 1]:.1f}" for i in range(len(confidence_bins) - 1)]

        confidence_analysis = {}

        if verbose:
            print("Confidence Level vs Error Rate:")
            print("-" * 40)

        for i in range(len(confidence_bins) - 1):
            bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba < confidence_bins[i + 1])
            if i == len(confidence_bins) - 2:  # Last bin includes 1.0
                bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba <= confidence_bins[i + 1])

            if np.sum(bin_mask) > 0:
                bin_error_rate = np.mean(errors[bin_mask])
                bin_count = np.sum(bin_mask)

                confidence_analysis[bin_labels[i]] = {
                    'error_rate': bin_error_rate,
                    'count': bin_count
                }

                if verbose:
                    print(f"  {bin_labels[i]:8} | Error Rate: {bin_error_rate:.3f} | Count: {bin_count:3d}")

        if verbose:
            print(f"\n💡 IDEAL PATTERN: Error rate should decrease as confidence increases")

            # Check if pattern is monotonic
            error_rates = [confidence_analysis[label]['error_rate'] for label in bin_labels
                           if label in confidence_analysis and confidence_analysis[label]['count'] > 5]

            if len(error_rates) > 2:
                is_decreasing = all(error_rates[i] >= error_rates[i + 1] for i in range(len(error_rates) - 1))
                if is_decreasing:
                    print("✅ Good calibration: Higher confidence → Lower error rate")
                else:
                    print("⚠️  Poor calibration: Confidence doesn't match performance")

        return confidence_analysis

    def _analyze_feature_errors(self, X_test_orig, errors, verbose=True):
        """Analyze which features are associated with errors"""

        feature_error_analysis = {}

        # For categorical features
        categorical_cols = X_test_orig.select_dtypes(include=['object']).columns

        if verbose and len(categorical_cols) > 0:
            print("Feature values most associated with errors:")
            print("-" * 50)

        for col in categorical_cols:
            feature_errors = {}
            overall_error_rate = np.mean(errors)

            for value in X_test_orig[col].unique():
                mask = X_test_orig[col] == value
                if np.sum(mask) >= 5:  # Only analyze if sufficient samples
                    value_error_rate = np.mean(errors[mask])
                    relative_risk = value_error_rate / overall_error_rate if overall_error_rate > 0 else 1

                    feature_errors[value] = {
                        'error_rate': value_error_rate,
                        'relative_risk': relative_risk,
                        'count': np.sum(mask)
                    }

            # Find most problematic values
            if feature_errors:
                worst_value = max(feature_errors.keys(), key=lambda x: feature_errors[x]['relative_risk'])
                best_value = min(feature_errors.keys(), key=lambda x: feature_errors[x]['relative_risk'])

                feature_error_analysis[col] = {
                    'worst_value': worst_value,
                    'best_value': best_value,
                    'worst_relative_risk': feature_errors[worst_value]['relative_risk'],
                    'best_relative_risk': feature_errors[best_value]['relative_risk']
                }

                if verbose:
                    print(f"\n{col}:")
                    print(f"  Most problematic: {worst_value} (RR: {feature_errors[worst_value]['relative_risk']:.2f})")
                    print(f"  Best performing:  {best_value} (RR: {feature_errors[best_value]['relative_risk']:.2f})")

        return feature_error_analysis

    def _provide_error_recommendations(self, segment_analysis, error_predictability,
                                       pattern_analysis, special_values_analysis, verbose=True):
        """Provide actionable recommendations based on error analysis"""

        if not verbose:
            return

        print("🎯 ACTIONABLE RECOMMENDATIONS:")
        print("-" * 40)

        # Weak segments recommendations
        if segment_analysis:
            print("1. SEGMENT-SPECIFIC IMPROVEMENTS:")
            print("   • Focus on retraining with more data from weak segments")
            print("   • Consider segment-specific models or features")
            print("   • Implement different decision thresholds per segment")

        # Error predictability recommendations
        if error_predictability.get('is_predictable', False):
            print("\n2. ERROR PREDICTION STRATEGY:")
            print("   • Implement prediction confidence scoring")
            print("   • Flag high-risk predictions for manual review")
            print("   • Use ensemble methods to reduce predictable errors")

        # Confidence calibration recommendations
        if 'confidence' in pattern_analysis:
            error_conf = pattern_analysis['confidence']['error_mean_confidence']
            if error_conf > 0.6:
                print("\n3. CONFIDENCE CALIBRATION:")
                print("   • Model is overconfident - implement calibration")
                print("   • Consider Platt scaling or isotonic regression")
                print("   • Add uncertainty quantification")

        # Special values recommendations
        if special_values_analysis:
            print("\n4. DATA QUALITY IMPROVEMENTS:")
            print("   • Review data preprocessing for outliers")
            print("   • Consider different imputation strategies")
            print("   • Implement outlier detection in production")

        print("\n5. GENERAL RECOMMENDATIONS:")
        print("   • Collect more training data for weak segments")
        print("   • Implement A/B testing for model improvements")
        print("   • Set up continuous monitoring for error patterns")
        print("   • Consider cost-sensitive learning for business impact")
        """Perform residual analysis for the model"""

        if model_name not in self.ml_pipeline.models:
            print(f"Model '{model_name}' not found")
            return

        model = self.ml_pipeline.models[model_name]

        # Use validation split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.ml_pipeline.X_train_processed,
            self.ml_pipeline.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.ml_pipeline.y_train
        )

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate residuals (for probability predictions)
        residuals = y_test - y_pred_proba

        print("=" * 50)
        print(f"RESIDUAL ANALYSIS: {model_name}")
        print("=" * 50)

        print(f"Mean Residual:      {np.mean(residuals):.6f}")
        print(f"Std Residual:       {np.std(residuals):.6f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.6f}")
        print(f"RMSE:               {np.sqrt(np.mean(residuals ** 2)):.6f}")

        # Plot residuals
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Residuals vs Predicted
        axes[0].scatter(y_pred_proba, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot of Residuals')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def comprehensive_error_analysis(self, model_name, verbose=True):
        """
        Comprehensive error analysis answering key questions:
        - Do we have weak segments?
        - Can we predict the error?
        - Any patterns?
        - Special values?
        """

        if model_name not in self.ml_pipeline.models:
            print(f"Model '{model_name}' not found")
            return

        model = self.ml_pipeline.models[model_name]

        # Prepare test data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.ml_pipeline.X_train_processed,
            self.ml_pipeline.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.ml_pipeline.y_train
        )

        # Get original feature data for analysis
        X_train_orig, X_test_orig, _, _ = train_test_split(
            self.ml_pipeline.X_train,
            self.ml_pipeline.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.ml_pipeline.y_train
        )

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Create error flags
        errors = (y_pred != y_test)
        correct_predictions = ~errors

        print("=" * 80)
        print(f"COMPREHENSIVE ERROR ANALYSIS: {model_name}")
        print("=" * 80)

        # Overall Error Statistics
        total_errors = np.sum(errors)
        total_predictions = len(y_test)
        error_rate = total_errors / total_predictions

        print(f"\n📊 OVERALL ERROR STATISTICS")
        print("-" * 50)
        print(f"Total Predictions:        {total_predictions}")
        print(f"Total Errors:             {total_errors}")
        print(f"Overall Error Rate:       {error_rate:.4f} ({error_rate * 100:.2f}%)")
        print(f"Accuracy:                 {1 - error_rate:.4f} ({(1 - error_rate) * 100:.2f}%)")

        # Error Type Analysis
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        false_negatives = np.sum((y_pred == 0) & (y_test == 1))

        print(f"\n❌ ERROR TYPE BREAKDOWN")
        print("-" * 50)
        print(f"False Positives (Type I):  {false_positives} ({false_positives / total_predictions * 100:.2f}%)")
        print(f"  → Incorrectly approved loans that should be rejected")
        print(f"False Negatives (Type II): {false_negatives} ({false_negatives / total_predictions * 100:.2f}%)")
        print(f"  → Incorrectly rejected loans that should be approved")

        if verbose:
            print(f"\n💡 BUSINESS IMPACT INTERPRETATION:")
            print(f"  • False Positives = Potential loan defaults ($$ RISK)")
            print(f"  • False Negatives = Lost business opportunities ($ LOSS)")
            if false_positives > false_negatives:
                print(f"  ⚠️  Model is more risky - approving bad loans!")
            elif false_negatives > false_positives:
                print(f"  ⚠️  Model is too conservative - rejecting good loans!")
            else:
                print(f"  ✅ Model has balanced error types")

        # 1. WEAK SEGMENTS ANALYSIS
        print(f"\n🎯 1. WEAK SEGMENTS ANALYSIS")
        print("=" * 50)

        segment_analysis = self._analyze_weak_segments(X_test_orig, y_test, y_pred, errors, verbose)

        # 2. ERROR PREDICTABILITY ANALYSIS
        print(f"\n🔮 2. CAN WE PREDICT THE ERROR?")
        print("=" * 50)

        error_predictability = self._analyze_error_predictability(X_test, y_test, y_pred, y_pred_proba, errors, verbose)

        # 3. ERROR PATTERNS ANALYSIS
        print(f"\n🔍 3. ERROR PATTERNS ANALYSIS")
        print("=" * 50)

        pattern_analysis = self._analyze_error_patterns(X_test_orig, y_test, y_pred, y_pred_proba, errors, verbose)

        # 4. SPECIAL VALUES ANALYSIS
        print(f"\n⭐ 4. SPECIAL VALUES ANALYSIS")
        print("=" * 50)

        special_values_analysis = self._analyze_special_values(X_test_orig, y_test, y_pred, errors, verbose)

        # 5. CONFIDENCE-ERROR RELATIONSHIP
        print(f"\n📈 5. CONFIDENCE vs ERROR RELATIONSHIP")
        print("=" * 50)

        confidence_analysis = self._analyze_confidence_errors(y_pred_proba, errors, verbose)

        # 6. FEATURE-SPECIFIC ERROR ANALYSIS
        print(f"\n🔧 6. FEATURE-SPECIFIC ERROR ANALYSIS")
        print("=" * 50)

        feature_error_analysis = self._analyze_feature_errors(X_test_orig, errors, verbose)

        # Summary and Recommendations
        print(f"\n🎯 SUMMARY & RECOMMENDATIONS")
        print("=" * 50)
        self._provide_error_recommendations(segment_analysis, error_predictability,
                                            pattern_analysis, special_values_analysis, verbose)

        return {
            'segment_analysis': segment_analysis,
            'error_predictability': error_predictability,
            'pattern_analysis': pattern_analysis,
            'special_values_analysis': special_values_analysis,
            'confidence_analysis': confidence_analysis,
            'feature_error_analysis': feature_error_analysis
        }

    def _analyze_weak_segments(self, X_test_orig, y_test, y_pred, errors, verbose=True):
        """Analyze which segments have high error rates"""

        segment_results = {}

        # Analyze categorical variables
        categorical_cols = X_test_orig.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if verbose:
                print(f"\n📊 Segment Analysis: {col}")
                print("-" * 30)

            segment_errors = {}
            for value in X_test_orig[col].unique():
                mask = X_test_orig[col] == value
                segment_error_rate = np.mean(errors[mask])
                segment_size = np.sum(mask)
                segment_errors[value] = {
                    'error_rate': segment_error_rate,
                    'size': segment_size,
                    'total_errors': np.sum(errors[mask])
                }

                if verbose:
                    print(
                        f"  {value:15} | Error Rate: {segment_error_rate:.3f} | Size: {segment_size:3d} | Errors: {np.sum(errors[mask]):2d}")

            # Find weakest segment
            weakest_segment = max(segment_errors.keys(), key=lambda x: segment_errors[x]['error_rate'])
            strongest_segment = min(segment_errors.keys(), key=lambda x: segment_errors[x]['error_rate'])

            if verbose:
                print(
                    f"  🔴 WEAKEST:  {weakest_segment} ({segment_errors[weakest_segment]['error_rate']:.3f} error rate)")
                print(
                    f"  🟢 STRONGEST: {strongest_segment} ({segment_errors[strongest_segment]['error_rate']:.3f} error rate)")

            segment_results[col] = segment_errors

        # Analyze numerical variables (quartiles)
        numerical_cols = X_test_orig.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if verbose:
                print(f"\n📊 Segment Analysis: {col} (Quartiles)")
                print("-" * 30)

            quartiles = np.percentile(X_test_orig[col], [25, 50, 75])
            quartile_labels = ['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)']
            quartile_errors = {}

            for i, label in enumerate(quartile_labels):
                if i == 0:
                    mask = X_test_orig[col] <= quartiles[0]
                elif i == 1:
                    mask = (X_test_orig[col] > quartiles[0]) & (X_test_orig[col] <= quartiles[1])
                elif i == 2:
                    mask = (X_test_orig[col] > quartiles[1]) & (X_test_orig[col] <= quartiles[2])
                else:
                    mask = X_test_orig[col] > quartiles[2]

                segment_error_rate = np.mean(errors[mask])
                segment_size = np.sum(mask)
                quartile_errors[label] = {
                    'error_rate': segment_error_rate,
                    'size': segment_size,
                    'total_errors': np.sum(errors[mask])
                }

                if verbose:
                    print(
                        f"  {label:12} | Error Rate: {segment_error_rate:.3f} | Size: {segment_size:3d} | Errors: {np.sum(errors[mask]):2d}")

            segment_results[f"{col}_quartiles"] = quartile_errors

        return segment_results

    def _analyze_error_predictability(self, X_test, y_test, y_pred, y_pred_proba, errors, verbose=True):
        """Analyze if we can predict where errors will occur"""

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Train a model to predict errors
        try:
            error_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            error_predictor.fit(X_test, errors)

            error_pred = error_predictor.predict(X_test)
            error_pred_proba = error_predictor.predict_proba(X_test)[:, 1]

            error_prediction_accuracy = accuracy_score(errors, error_pred)

            if len(np.unique(errors)) > 1:  # Check if we have both classes
                error_prediction_auc = roc_auc_score(errors, error_pred_proba)
            else:
                error_prediction_auc = 0.5

            if verbose:
                print(f"Error Prediction Accuracy:    {error_prediction_accuracy:.4f}")
                print(f"Error Prediction AUC:         {error_prediction_auc:.4f}")

                if error_prediction_auc > 0.7:
                    print("✅ ERRORS ARE HIGHLY PREDICTABLE!")
                    print("   → We can identify high-risk predictions")
                    print("   → Consider using prediction confidence thresholds")
                elif error_prediction_auc > 0.6:
                    print("⚠️  ERRORS ARE SOMEWHAT PREDICTABLE")
                    print("   → Some patterns exist but not very strong")
                else:
                    print("❌ ERRORS APPEAR RANDOM")
                    print("   → No clear pattern in where errors occur")

            # Feature importance for error prediction
            feature_importance = error_predictor.feature_importances_
            feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]

            if hasattr(self.ml_pipeline, 'X_train') and hasattr(self.ml_pipeline.X_train, 'columns'):
                feature_names = self.ml_pipeline.X_train.columns

            top_error_features = sorted(zip(feature_names, feature_importance),
                                        key=lambda x: x[1], reverse=True)[:5]

            if verbose:
                print(f"\nTop 5 features that predict errors:")
                for i, (feature, importance) in enumerate(top_error_features, 1):
                    print(f"  {i}. {feature}: {importance:.4f}")

            return {
                'predictability_accuracy': error_prediction_accuracy,
                'predictability_auc': error_prediction_auc,
                'top_error_features': top_error_features,
                'is_predictable': error_prediction_auc > 0.6
            }

        except Exception as e:
            if verbose:
                print(f"Could not analyze error predictability: {e}")
            return {'predictability_accuracy': 0, 'predictability_auc': 0.5, 'is_predictable': False}

    def _analyze_error_patterns(self, X_test_orig, y_test, y_pred, y_pred_proba, errors, verbose=True):
        """Analyze patterns in errors"""

        patterns = {}

        # Confidence patterns
        error_confidence = y_pred_proba[errors]
        correct_confidence = y_pred_proba[~errors]

        patterns['confidence'] = {
            'error_mean_confidence': np.mean(error_confidence),
            'correct_mean_confidence': np.mean(correct_confidence),
            'error_std_confidence': np.std(error_confidence),
            'correct_std_confidence': np.std(correct_confidence)
        }

        if verbose:
            print(f"Average confidence for ERRORS:     {np.mean(error_confidence):.4f}")
            print(f"Average confidence for CORRECT:    {np.mean(correct_confidence):.4f}")
            print(f"Confidence std for ERRORS:         {np.std(error_confidence):.4f}")
            print(f"Confidence std for CORRECT:        {np.std(correct_confidence):.4f}")

            if np.mean(error_confidence) < 0.6:
                print("✅ Good! Errors tend to have lower confidence")
                print("   → Model is appropriately uncertain about wrong predictions")
            else:
                print("⚠️  Concerning! Errors have high confidence")
                print("   → Model is overconfident in wrong predictions")

        # Class-specific error patterns
        false_positive_mask = (y_pred == 1) & (y_test == 0)
        false_negative_mask = (y_pred == 0) & (y_test == 1)

        if np.sum(false_positive_mask) > 0:
            fp_confidence = y_pred_proba[false_positive_mask]
            patterns['false_positive_confidence'] = {
                'mean': np.mean(fp_confidence),
                'std': np.std(fp_confidence)
            }

            if verbose:
                print(f"\nFalse Positive Analysis:")
                print(f"  Average confidence: {np.mean(fp_confidence):.4f}")
                print(f"  → Model confidence when incorrectly approving loans")

        if np.sum(false_negative_mask) > 0:
            fn_confidence = y_pred_proba[false_negative_mask]
            patterns['false_negative_confidence'] = {
                'mean': np.mean(fn_confidence),
                'std': np.std(fn_confidence)
            }

            if verbose:
                print(f"\nFalse Negative Analysis:")
                print(f"  Average confidence: {np.mean(fn_confidence):.4f}")
                print(f"  → Model confidence when incorrectly rejecting loans")

        return patterns

    def _analyze_special_values(self, X_test_orig, y_test, y_pred, errors, verbose=True):
        """Analyze errors related to special values"""

        special_analysis = {}

        # Check for missing value patterns (after imputation, look for imputed values)
        if verbose:
            print("Looking for special value patterns...")

        numerical_cols = X_test_orig.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            col_data = X_test_orig[col]

            # Check for values at quartiles (potential imputed values)
            median_val = np.median(col_data)
            mean_val = np.mean(col_data)

            # Values exactly at median (possible imputation)
            median_mask = np.abs(col_data - median_val) < 1e-10
            if np.sum(median_mask) > len(col_data) * 0.05:  # More than 5% at exact median
                median_error_rate = np.mean(errors[median_mask])
                other_error_rate = np.mean(errors[~median_mask])

                special_analysis[f"{col}_median_values"] = {
                    'count': np.sum(median_mask),
                    'error_rate': median_error_rate,
                    'other_error_rate': other_error_rate
                }

                if verbose:
                    print(f"\n🔍 {col} - Potential imputed values at median:")
                    print(f"  Count at median: {np.sum(median_mask)}")
                    print(f"  Error rate at median: {median_error_rate:.4f}")
                    print(f"  Error rate elsewhere: {other_error_rate:.4f}")

            # Check for outliers
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            outlier_mask = (col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)

            if np.sum(outlier_mask) > 0:
                outlier_error_rate = np.mean(errors[outlier_mask])
                normal_error_rate = np.mean(errors[~outlier_mask])

                special_analysis[f"{col}_outliers"] = {
                    'count': np.sum(outlier_mask),
                    'error_rate': outlier_error_rate,
                    'normal_error_rate': normal_error_rate
                }

                if verbose:
                    print(f"\n📊 {col} - Outlier analysis:")
                    print(f"  Outlier count: {np.sum(outlier_mask)}")
                    print(f"  Error rate for outliers: {outlier_error_rate:.4f}")
                    print(f"  Error rate for normal values: {normal_error_rate:.4f}")

                    if outlier_error_rate > normal_error_rate * 1.5:
                        print(f"  ⚠️  OUTLIERS HAVE MUCH HIGHER ERROR RATE!")
                    elif outlier_error_rate < normal_error_rate * 0.5:
                        print(f"  ✅ Outliers actually perform better")

        return special_analysis

    def _analyze_confidence_errors(self, y_pred_proba, errors, verbose=True):
        """Analyze relationship between prediction confidence and errors"""

        # Bin by confidence levels
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_labels = [f"{confidence_bins[i]:.1f}-{confidence_bins[i + 1]:.1f}" for i in range(len(confidence_bins) - 1)]

        confidence_analysis = {}

        if verbose:
            print("Confidence Level vs Error Rate:")
            print("-" * 40)

        for i in range(len(confidence_bins) - 1):
            bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba < confidence_bins[i + 1])
            if i == len(confidence_bins) - 2:  # Last bin includes 1.0
                bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba <= confidence_bins[i + 1])

            if np.sum(bin_mask) > 0:
                bin_error_rate = np.mean(errors[bin_mask])
                bin_count = np.sum(bin_mask)

                confidence_analysis[bin_labels[i]] = {
                    'error_rate': bin_error_rate,
                    'count': bin_count
                }

                if verbose:
                    print(f"  {bin_labels[i]:8} | Error Rate: {bin_error_rate:.3f} | Count: {bin_count:3d}")

        if verbose:
            print(f"\n💡 IDEAL PATTERN: Error rate should decrease as confidence increases")

            # Check if pattern is monotonic
            error_rates = [confidence_analysis[label]['error_rate'] for label in bin_labels
                           if label in confidence_analysis and confidence_analysis[label]['count'] > 5]

            if len(error_rates) > 2:
                is_decreasing = all(error_rates[i] >= error_rates[i + 1] for i in range(len(error_rates) - 1))
                if is_decreasing:
                    print("✅ Good calibration: Higher confidence → Lower error rate")
                else:
                    print("⚠️  Poor calibration: Confidence doesn't match performance")

        return confidence_analysis

    def _analyze_feature_errors(self, X_test_orig, errors, verbose=True):
        """Analyze which features are associated with errors"""

        feature_error_analysis = {}

        # For categorical features
        categorical_cols = X_test_orig.select_dtypes(include=['object']).columns

        if verbose and len(categorical_cols) > 0:
            print("Feature values most associated with errors:")
            print("-" * 50)

        for col in categorical_cols:
            feature_errors = {}
            overall_error_rate = np.mean(errors)

            for value in X_test_orig[col].unique():
                mask = X_test_orig[col] == value
                if np.sum(mask) >= 5:  # Only analyze if sufficient samples
                    value_error_rate = np.mean(errors[mask])
                    relative_risk = value_error_rate / overall_error_rate if overall_error_rate > 0 else 1

                    feature_errors[value] = {
                        'error_rate': value_error_rate,
                        'relative_risk': relative_risk,
                        'count': np.sum(mask)
                    }

            # Find most problematic values
            if feature_errors:
                worst_value = max(feature_errors.keys(), key=lambda x: feature_errors[x]['relative_risk'])
                best_value = min(feature_errors.keys(), key=lambda x: feature_errors[x]['relative_risk'])

                feature_error_analysis[col] = {
                    'worst_value': worst_value,
                    'best_value': best_value,
                    'worst_relative_risk': feature_errors[worst_value]['relative_risk'],
                    'best_relative_risk': feature_errors[best_value]['relative_risk']
                }

                if verbose:
                    print(f"\n{col}:")
                    print(f"  Most problematic: {worst_value} (RR: {feature_errors[worst_value]['relative_risk']:.2f})")
                    print(f"  Best performing:  {best_value} (RR: {feature_errors[best_value]['relative_risk']:.2f})")

        return feature_error_analysis

    def _provide_error_recommendations(self, segment_analysis, error_predictability,
                                       pattern_analysis, special_values_analysis, verbose=True):
        """Provide actionable recommendations based on error analysis"""

        if not verbose:
            return

        print("🎯 ACTIONABLE RECOMMENDATIONS:")
        print("-" * 40)

        # Weak segments recommendations
        if segment_analysis:
            print("1. SEGMENT-SPECIFIC IMPROVEMENTS:")
            print("   • Focus on retraining with more data from weak segments")
            print("   • Consider segment-specific models or features")
            print("   • Implement different decision thresholds per segment")

        # Error predictability recommendations
        if error_predictability.get('is_predictable', False):
            print("\n2. ERROR PREDICTION STRATEGY:")
            print("   • Implement prediction confidence scoring")
            print("   • Flag high-risk predictions for manual review")
            print("   • Use ensemble methods to reduce predictable errors")

        # Confidence calibration recommendations
        if 'confidence' in pattern_analysis:
            error_conf = pattern_analysis['confidence']['error_mean_confidence']
            if error_conf > 0.6:
                print("\n3. CONFIDENCE CALIBRATION:")
                print("   • Model is overconfident - implement calibration")
                print("   • Consider Platt scaling or isotonic regression")
                print("   • Add uncertainty quantification")

        # Special values recommendations
        if special_values_analysis:
            print("\n4. DATA QUALITY IMPROVEMENTS:")
            print("   • Review data preprocessing for outliers")
            print("   • Consider different imputation strategies")
            print("   • Implement outlier detection in production")

        print("\n5. GENERAL RECOMMENDATIONS:")
        print("   • Collect more training data for weak segments")
        print("   • Implement A/B testing for model improvements")
        print("   • Set up continuous monitoring for error patterns")
        print("   • Consider cost-sensitive learning for business impact")

    def comprehensive_analysis_all_models(self, verbose=True, generate_plots=True):
        """
        Run comprehensive goodness of fit analysis for ALL models
        """

        if not self.ml_pipeline.models:
            print("No models found in the pipeline!")
            return

        print("=" * 100)
        print("COMPREHENSIVE GOODNESS OF FIT ANALYSIS - ALL MODELS")
        print("=" * 100)

        all_results = {}

        # Get all model names
        model_names = list(self.ml_pipeline.models.keys())

        print(f"\nFound {len(model_names)} models: {', '.join(model_names)}")
        print("\nRunning comprehensive analysis for each model...")

        for i, model_name in enumerate(model_names, 1):
            print(f"\n{'=' * 60}")
            print(f"ANALYZING MODEL {i}/{len(model_names)}: {model_name}")
            print(f"{'=' * 60}")

            try:
                # 1. Basic evaluation
                if verbose:
                    print(f"\n🔍 Step 1: Basic Metrics Evaluation")
                basic_results = self.evaluate_model(model_name)

                # 2. Display metrics summary
                if verbose:
                    print(f"\n📊 Step 2: Detailed Metrics Summary")
                self.display_metrics_summary(model_name)

                # 3. Comprehensive error analysis
                if verbose:
                    print(f"\n🔬 Step 3: Comprehensive Error Analysis")
                error_results = self.comprehensive_error_analysis(model_name, verbose=verbose)

                # 4. Generate plots if requested
                if generate_plots:
                    if verbose:
                        print(f"\n📈 Step 4: Generating Comprehensive Plots")
                    self.plot_comprehensive_evaluation(model_name)

                # Store all results
                all_results[model_name] = {
                    'basic_metrics': basic_results,
                    'error_analysis': error_results
                }

                print(f"\n✅ Completed analysis for {model_name}")

            except Exception as e:
                print(f"\n❌ Error analyzing {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}

        # Final comparison and summary
        print(f"\n{'=' * 100}")
        print("FINAL MODEL COMPARISON & RECOMMENDATIONS")
        print(f"{'=' * 100}")

        # Model comparison
        print(f"\n🏆 OVERALL MODEL COMPARISON:")
        comparison_df = self.compare_models(model_names)

        # Best model recommendation
        self._recommend_best_model(comparison_df, all_results, verbose)

        # Generate summary report
        if verbose:
            self._generate_summary_report(all_results, comparison_df)

        return all_results

    def _recommend_best_model(self, comparison_df, all_results, verbose=True):
        """Recommend the best model based on comprehensive analysis"""

        if comparison_df is None or len(comparison_df) == 0:
            return

        print(f"\n🎯 MODEL RECOMMENDATION:")
        print("-" * 50)

        # Get top 3 models by ROC AUC
        top_models = comparison_df.head(3)

        print(f"🥇 BEST OVERALL: {top_models.iloc[0]['Model']}")
        print(f"   ROC AUC: {top_models.iloc[0]['roc_auc']:.4f}")
        print(f"   Accuracy: {top_models.iloc[0]['accuracy']:.4f}")
        print(f"   F1-Score: {top_models.iloc[0]['f1_score']:.4f}")

        if len(top_models) > 1:
            print(f"\n🥈 RUNNER-UP: {top_models.iloc[1]['Model']}")
            print(f"   ROC AUC: {top_models.iloc[1]['roc_auc']:.4f}")

        if len(top_models) > 2:
            print(f"\n🥉 THIRD PLACE: {top_models.iloc[2]['Model']}")
            print(f"   ROC AUC: {top_models.iloc[2]['roc_auc']:.4f}")

        # Provide specific recommendations
        best_model_name = top_models.iloc[0]['Model']

        print(f"\n💡 DEPLOYMENT RECOMMENDATION:")
        print(f"   → Deploy: {best_model_name}")

        # Check if there are close competitors
        if len(top_models) > 1:
            auc_diff = top_models.iloc[0]['roc_auc'] - top_models.iloc[1]['roc_auc']
            if auc_diff < 0.02:
                print(f"   ⚠️  Close competition with {top_models.iloc[1]['Model']} (diff: {auc_diff:.4f})")
                print(f"   → Consider ensemble of top models")

        # Business considerations
        print(f"\n💼 BUSINESS CONSIDERATIONS:")

        # Check error types from the best model
        if best_model_name in all_results and 'basic_metrics' in all_results[best_model_name]:
            metrics = all_results[best_model_name]['basic_metrics']

            if 'false_positive' in metrics and 'false_negative' in metrics:
                fp_rate = metrics['false_positive'] / (metrics['false_positive'] + metrics['true_negative'])
                fn_rate = metrics['false_negative'] / (metrics['false_negative'] + metrics['true_positive'])

                print(f"   False Positive Rate: {fp_rate:.3f} (Risk: Approving bad loans)")
                print(f"   False Negative Rate: {fn_rate:.3f} (Risk: Rejecting good loans)")

                if fp_rate > fn_rate * 1.5:
                    print(f"   ⚠️  Model is risky - consider higher approval threshold")
                elif fn_rate > fp_rate * 1.5:
                    print(f"   ⚠️  Model is conservative - consider lower approval threshold")

    def _generate_summary_report(self, all_results, comparison_df):
        """Generate a comprehensive summary report"""

        print(f"\n📋 EXECUTIVE SUMMARY REPORT:")
        print("=" * 60)

        successful_models = [name for name, result in all_results.items() if 'error' not in result]
        failed_models = [name for name, result in all_results.items() if 'error' in result]

        print(f"✅ Successfully analyzed: {len(successful_models)} models")
        if failed_models:
            print(f"❌ Failed to analyze: {len(failed_models)} models ({', '.join(failed_models)})")

        if comparison_df is not None and len(comparison_df) > 0:
            # Performance statistics
            print(f"\n📊 PERFORMANCE STATISTICS:")
            print(f"   Best Accuracy:     {comparison_df['accuracy'].max():.4f}")
            print(f"   Worst Accuracy:    {comparison_df['accuracy'].min():.4f}")
            print(f"   Best ROC AUC:      {comparison_df['roc_auc'].max():.4f}")
            print(f"   Worst ROC AUC:     {comparison_df['roc_auc'].min():.4f}")
            print(f"   Best F1-Score:     {comparison_df['f1_score'].max():.4f}")
            print(f"   Performance Range: {comparison_df['roc_auc'].max() - comparison_df['roc_auc'].min():.4f}")

        # Error analysis summary
        print(f"\n🔍 ERROR ANALYSIS INSIGHTS:")
        error_predictable_count = 0
        high_confidence_error_count = 0

        for model_name, results in all_results.items():
            if 'error_analysis' in results:
                error_analysis = results['error_analysis']

                # Count predictable errors
                if error_analysis.get('error_predictability', {}).get('is_predictable', False):
                    error_predictable_count += 1

                # Count high confidence errors
                if 'pattern_analysis' in error_analysis:
                    confidence_data = error_analysis['pattern_analysis'].get('confidence', {})
                    if confidence_data.get('error_mean_confidence', 0) > 0.6:
                        high_confidence_error_count += 1

        if successful_models:
            print(f"   Models with predictable errors: {error_predictable_count}/{len(successful_models)}")
            print(f"   Models with overconfident errors: {high_confidence_error_count}/{len(successful_models)}")

        # Recommendations summary
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        print(
            f"   1. Use {comparison_df.iloc[0]['Model'] if comparison_df is not None and len(comparison_df) > 0 else 'best performing model'} for production")
        print(f"   2. Implement prediction confidence thresholds")
        print(f"   3. Monitor weak segments identified in error analysis")
        print(f"   4. Consider ensemble methods if models perform similarly")
        print(f"   5. Set up continuous model monitoring in production")

    def quick_model_overview(self):
        """Quick overview of all models without detailed analysis"""

        print("=" * 80)
        print("QUICK MODEL OVERVIEW")
        print("=" * 80)

        if not self.ml_pipeline.models:
            print("No models found!")
            return

        # Evaluate all models quickly
        quick_results = []

        for model_name in self.ml_pipeline.models.keys():
            try:
                if model_name not in self.results:
                    self.evaluate_model(model_name)

                result = self.results[model_name]
                quick_results.append({
                    'Model': model_name,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'ROC_AUC': f"{result['roc_auc']:.4f}",
                    'F1_Score': f"{result['f1_score']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}"
                })
            except Exception as e:
                quick_results.append({
                    'Model': model_name,
                    'Status': f'Error: {str(e)[:50]}...'
                })

        # Display results
        import pandas as pd
        overview_df = pd.DataFrame(quick_results)

        if 'ROC_AUC' in overview_df.columns:
            # Sort by ROC AUC if available
            overview_df = overview_df.sort_values('ROC_AUC', ascending=False)

        print(overview_df.to_string(index=False))

        # Quick recommendations
        if 'ROC_AUC' in overview_df.columns:
            best_model = overview_df.iloc[0]['Model']
            best_auc = overview_df.iloc[0]['ROC_AUC']
            print(f"\n🏆 Best Model: {best_model} (ROC AUC: {best_auc})")

        return overview_df

    def save_analysis_report(self, filename='loan_model_analysis_report.txt'):
        """Save comprehensive analysis to a text file"""

        import sys
        from io import StringIO

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Run comprehensive analysis
            self.comprehensive_analysis_all_models(verbose=True, generate_plots=False)
        finally:
            sys.stdout = old_stdout

        # Save to file
        report_content = captured_output.getvalue()

        with open(filename, 'w') as f:
            f.write("LOAN PREDICTION MODEL - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_content)

        print(f"📄 Analysis report saved to: {filename}")
        return filename

# Usage Examples for All Models Analysis:
# Initialize goodness of fit evaluator
# gof = GoodnessOfFit(ml_pipeline)
#
# # Evaluate specific model
# gof.evaluate_model('Random Forest')
# gof.display_metrics_summary('Random Forest')
# gof.plot_comprehensive_evaluation('Random Forest')
#
# # Compare all models
# gof.compare_models()
#
# # Residual analysis
# gof.residual_analysis('Random Forest')