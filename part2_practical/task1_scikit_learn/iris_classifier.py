"""
Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal: Preprocess data, train decision tree classifier, evaluate performance

Author: [Your Team Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """
    A comprehensive Iris species classifier using Decision Tree algorithm.
    """
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.model = DecisionTreeClassifier(random_state=42, max_depth=5)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def load_and_explore_data(self):
        """
        Load the Iris dataset and perform exploratory data analysis.
        
        Returns:
            tuple: (X, y, feature_names, target_names, df)
        """
        print("=" * 50)
        print("LOADING AND EXPLORING IRIS DATASET")
        print("=" * 50)
        
        # Load the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Create a pandas DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = [target_names[i] for i in y]
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {feature_names}")
        print(f"Target classes: {target_names}")
        print(f"\\nDataset info:")
        print(df.info())
        print(f"\\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\\nMissing values:")
        print(df.isnull().sum())
        
        # Basic statistics
        print(f"\\nBasic statistics:")
        print(df.describe())
        
        # Class distribution
        print(f"\\nClass distribution:")
        print(df['species'].value_counts())
        
        return X, y, feature_names, target_names, df
    
    def visualize_data(self, df, feature_names):
        """
        Create visualizations to understand the data better.
        
        Args:
            df (pd.DataFrame): The iris dataframe
            feature_names (list): List of feature names
        """
        print("\\n" + "=" * 50)
        print("DATA VISUALIZATION")
        print("=" * 50)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Pairwise scatter plots
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                       hue='species', alpha=0.7)
        plt.title('Sepal Length vs Width')
        
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                       hue='species', alpha=0.7)
        plt.title('Petal Length vs Width')
        
        # 2. Distribution plots
        plt.subplot(2, 2, 3)
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            plt.hist(subset['petal length (cm)'], alpha=0.6, label=species, bins=15)
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Frequency')
        plt.title('Petal Length Distribution by Species')
        plt.legend()
        
        # 3. Correlation heatmap
        plt.subplot(2, 2, 4)
        correlation_matrix = df[feature_names].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Additional box plots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, feature in enumerate(feature_names):
            sns.boxplot(data=df, x='species', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} by Species')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, X, y):
        """
        Preprocess the data: handle missing values, encode labels, split data.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target vector
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Check for missing values (Iris dataset typically has none)
        print(f"Missing values in features: {np.isnan(X).sum()}")
        print(f"Missing values in target: {np.isnan(y).sum()}")
        
        # Handle missing values if any (using mean imputation)
        if np.isnan(X).any():
            print("Handling missing values using mean imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Optional: Feature scaling (not strictly necessary for Decision Trees)
        # X_train_scaled = self.scaler.fit_transform(X_train)
        # X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the Decision Tree classifier.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
        """
        print("\\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        # Train the model
        print("Training Decision Tree Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Display model parameters
        print(f"Model parameters:")
        print(f"- Max depth: {self.model.max_depth}")
        print(f"- Min samples split: {self.model.min_samples_split}")
        print(f"- Min samples leaf: {self.model.min_samples_leaf}")
        print(f"- Random state: {self.model.random_state}")
        
        # Feature importance
        feature_importance = self.model.feature_importance_
        print(f"\\nFeature Importance:")
        feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
        for name, importance in zip(feature_names, feature_importance):
            print(f"- {name}: {importance:.4f}")
    
    def evaluate_model(self, X_train, X_test, y_train, y_test, target_names):
        """
        Evaluate the model using various metrics.
        
        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
            target_names: Names of the target classes
        """
        print("\\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        if not self.is_trained:
            print("Error: Model must be trained first!")
            return
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification metrics
        print(f"\\nDetailed Classification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        # Precision and Recall for each class
        precision = precision_score(y_test, y_test_pred, average=None)
        recall = recall_score(y_test, y_test_pred, average=None)
        
        print(f"\\nPer-class Metrics:")
        for i, class_name in enumerate(target_names):
            print(f"- {class_name}:")
            print(f"  * Precision: {precision[i]:.4f}")
            print(f"  * Recall: {recall[i]:.4f}")
        
        # Overall averages
        avg_precision = precision_score(y_test, y_test_pred, average='weighted')
        avg_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        print(f"\\nWeighted Averages:")
        print(f"- Precision: {avg_precision:.4f}")
        print(f"- Recall: {avg_recall:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"\\nCross-validation Scores: {cv_scores}")
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'cv_scores': cv_scores
        }
    
    def visualize_decision_tree(self, feature_names, target_names):
        """
        Visualize the trained decision tree.
        
        Args:
            feature_names: Names of the features
            target_names: Names of the target classes
        """
        print("\\n" + "=" * 50)
        print("DECISION TREE VISUALIZATION")
        print("=" * 50)
        
        if not self.is_trained:
            print("Error: Model must be trained first!")
            return
        
        plt.figure(figsize=(20, 12))
        plot_tree(self.model, 
                 feature_names=feature_names,
                 class_names=target_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('Decision Tree for Iris Species Classification')
        plt.show()

def main():
    """
    Main function to run the complete Iris classification pipeline.
    """
    print("IRIS SPECIES CLASSIFICATION WITH SCIKIT-LEARN")
    print("=" * 60)
    
    # Initialize the classifier
    classifier = IrisClassifier()
    
    # Step 1: Load and explore the data
    X, y, feature_names, target_names, df = classifier.load_and_explore_data()
    
    # Step 2: Visualize the data
    classifier.visualize_data(df, feature_names)
    
    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    # Step 4: Train the model
    classifier.train_model(X_train, y_train)
    
    # Step 5: Evaluate the model
    results = classifier.evaluate_model(X_train, X_test, y_train, y_test, target_names)
    
    # Step 6: Visualize the decision tree
    classifier.visualize_decision_tree(feature_names, target_names)
    
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully trained Decision Tree Classifier")
    print(f"✅ Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"✅ Precision: {results['precision']:.4f}")
    print(f"✅ Recall: {results['recall']:.4f}")
    print(f"✅ CV Score: {results['cv_scores'].mean():.4f}")
    print("✅ All evaluation metrics computed successfully!")

if __name__ == "__main__":
    main()
