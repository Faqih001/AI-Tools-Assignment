"""
Task 2: Deep Learning with TensorFlow
Dataset: MNIST Handwritten Digits
Goal: Build CNN model, achieve >95% accuracy, visualize predictions

Author: [Your Team Name]
Date: [Current Date]
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MNISTClassifier:
    """
    A comprehensive MNIST digit classifier using Convolutional Neural Networks.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.history = None
        self.is_trained = False
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def load_and_explore_data(self):
        """
        Load and explore the MNIST dataset.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("=" * 50)
        print("LOADING AND EXPLORING MNIST DATASET")
        print("=" * 50)
        
        # Load the MNIST dataset
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Classes: {np.unique(y_train)}")
        
        # Display basic statistics
        print(f"\\nPixel value range: {X_train.min()} - {X_train.max()}")
        print(f"Data type: {X_train.dtype}")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\\nClass distribution in training set:")
        for digit, count in zip(unique, counts):
            print(f"Digit {digit}: {count} samples")
        
        return X_train, X_test, y_train, y_test
    
    def visualize_data(self, X_train, y_train):
        """
        Visualize sample images from the dataset.
        
        Args:
            X_train: Training images
            y_train: Training labels
        """
        print("\\n" + "=" * 50)
        print("DATA VISUALIZATION")
        print("=" * 50)
        
        # Plot sample images for each digit
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for digit in range(10):
            # Find first occurrence of each digit
            idx = np.where(y_train == digit)[0][0]
            axes[digit].imshow(X_train[idx], cmap='gray')
            axes[digit].set_title(f'Digit: {digit}')
            axes[digit].axis('off')
        
        plt.suptitle('Sample Images for Each Digit')
        plt.tight_layout()
        plt.show()
        
        # Plot more examples in a grid
        plt.figure(figsize=(12, 8))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(X_train[i], cmap='gray')
            plt.title(f'Label: {y_train[i]}')
            plt.axis('off')
        
        plt.suptitle('First 25 Training Images')
        plt.tight_layout()
        plt.show()
        
        # Pixel intensity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(X_train.flatten(), bins=50, alpha=0.7, color='blue')
        plt.title('Distribution of Pixel Intensities')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()
    
    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """
        Preprocess the data for CNN training.
        
        Args:
            X_train, X_test: Training and test images
            y_train, y_test: Training and test labels
            
        Returns:
            tuple: Preprocessed data
        """
        print("\\n" + "=" * 50)
        print("DATA PREPROCESSING")
        print("=" * 50)
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print("✅ Normalized pixel values to [0, 1]")
        
        # Reshape data to add channel dimension (for CNN)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        print("✅ Reshaped data for CNN input")
        
        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = keras.utils.to_categorical(y_train, 10)
        y_test_categorical = keras.utils.to_categorical(y_test, 10)
        print("✅ Converted labels to categorical format")
        
        print(f"\\nFinal shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train_categorical.shape}")
        print(f"y_test: {y_test_categorical.shape}")
        
        return X_train, X_test, y_train_categorical, y_test_categorical
    
    def build_model(self):
        """
        Build the CNN architecture.
        """
        print("\\n" + "=" * 50)
        print("BUILDING CNN MODEL")
        print("=" * 50)
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # Prevent overfitting
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Display model architecture
        print("Model Architecture:")
        self.model.summary()
        
        # Visualize model architecture
        keras.utils.plot_model(
            self.model, 
            to_file='model_architecture.png', 
            show_shapes=True, 
            show_layer_names=True
        )
        print("✅ Model architecture saved as 'model_architecture.png'")
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        """
        Train the CNN model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("\\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)
        
        if self.model is None:
            print("Error: Model must be built first!")
            return
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        print(f"Training parameters:")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Optimizer: Adam")
        print(f"- Loss function: Categorical Crossentropy")
        
        # Train the model
        print("\\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("✅ Training completed!")
    
    def evaluate_model(self, X_test, y_test, y_test_original):
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test images
            y_test: Test labels (categorical)
            y_test_original: Original test labels (integers)
        """
        print("\\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        if not self.is_trained:
            print("Error: Model must be trained first!")
            return
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Check if we achieved the target accuracy
        if test_accuracy > 0.95:
            print("🎉 SUCCESS: Achieved >95% test accuracy!")
        else:
            print("⚠️  WARNING: Did not achieve >95% test accuracy")
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Detailed classification report
        print(f"\\nDetailed Classification Report:")
        print(classification_report(y_test_original, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_original, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return test_accuracy, y_pred, y_pred_prob
    
    def plot_training_history(self):
        """
        Plot training history.
        """
        print("\\n" + "=" * 50)
        print("TRAINING HISTORY VISUALIZATION")
        print("=" * 50)
        
        if self.history is None:
            print("Error: No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print final metrics
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    def visualize_predictions(self, X_test, y_test_original, y_pred, y_pred_prob, num_samples=5):
        """
        Visualize model predictions on sample images.
        
        Args:
            X_test: Test images
            y_test_original: True labels
            y_pred: Predicted labels
            y_pred_prob: Prediction probabilities
            num_samples: Number of samples to visualize
        """
        print("\\n" + "=" * 50)
        print("PREDICTION VISUALIZATION")
        print("=" * 50)
        
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        
        for i, idx in enumerate(indices):
            # Original image
            axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            axes[0, i].set_title(f'True: {y_test_original[idx]}\\nPred: {y_pred[idx]}')
            axes[0, i].axis('off')
            
            # Prediction probabilities
            axes[1, i].bar(range(10), y_pred_prob[idx])
            axes[1, i].set_title(f'Confidence: {y_pred_prob[idx].max():.3f}')
            axes[1, i].set_xlabel('Digit')
            axes[1, i].set_ylabel('Probability')
            axes[1, i].set_xticks(range(10))
        
        plt.suptitle('Model Predictions on Sample Images')
        plt.tight_layout()
        plt.show()
        
        # Show some correct and incorrect predictions
        correct_indices = np.where(y_pred == y_test_original)[0][:5]
        incorrect_indices = np.where(y_pred != y_test_original)[0][:5]
        
        print(f"\\nCorrect Predictions Examples:")
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i, idx in enumerate(correct_indices):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {y_test_original[idx]}, Pred: {y_pred[idx]}')
            axes[i].axis('off')
        plt.suptitle('Correct Predictions')
        plt.show()
        
        if len(incorrect_indices) > 0:
            print(f"\\nIncorrect Predictions Examples:")
            fig, axes = plt.subplots(1, min(5, len(incorrect_indices)), figsize=(12, 3))
            if len(incorrect_indices) == 1:
                axes = [axes]
            for i, idx in enumerate(incorrect_indices[:5]):
                axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
                axes[i].set_title(f'True: {y_test_original[idx]}, Pred: {y_pred[idx]}')
                axes[i].axis('off')
            plt.suptitle('Incorrect Predictions')
            plt.show()
    
    def save_model(self, filepath='mnist_cnn_model.h5'):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"✅ Model saved to {filepath}")
        else:
            print("❌ No model to save!")

def main():
    """
    Main function to run the complete MNIST classification pipeline.
    """
    print("MNIST HANDWRITTEN DIGIT CLASSIFICATION WITH CNN")
    print("=" * 60)
    
    # Initialize the classifier
    classifier = MNISTClassifier()
    
    # Step 1: Load and explore the data
    X_train, X_test, y_train, y_test = classifier.load_and_explore_data()
    
    # Step 2: Visualize the data
    classifier.visualize_data(X_train, y_train)
    
    # Step 3: Preprocess the data
    X_train_processed, X_test_processed, y_train_cat, y_test_cat = classifier.preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # Step 4: Build the model
    classifier.build_model()
    
    # Step 5: Train the model
    classifier.train_model(X_train_processed, y_train_cat, X_test_processed, y_test_cat, epochs=15)
    
    # Step 6: Plot training history
    classifier.plot_training_history()
    
    # Step 7: Evaluate the model
    test_accuracy, y_pred, y_pred_prob = classifier.evaluate_model(
        X_test_processed, y_test_cat, y_test
    )
    
    # Step 8: Visualize predictions
    classifier.visualize_predictions(X_test_processed, y_test, y_pred, y_pred_prob)
    
    # Step 9: Save the model
    classifier.save_model()
    
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully trained CNN for MNIST classification")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    if test_accuracy > 0.95:
        print(f"🎉 TARGET ACHIEVED: >95% accuracy!")
    print(f"✅ Model saved successfully")
    print("✅ All visualizations generated!")

if __name__ == "__main__":
    main()
