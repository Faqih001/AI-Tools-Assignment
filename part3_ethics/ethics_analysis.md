# Part 3: Ethics & Optimization (10%)

## 1. Ethical Considerations

### Bias Identification in AI Models

Our assignment implementations include several models that may exhibit various types of bias. Here's a comprehensive analysis of potential biases and mitigation strategies:

---

### **MNIST Handwritten Digit Recognition Model**

#### **Potential Biases:**

1. **Data Collection Bias:**
   - The MNIST dataset was primarily collected from American Census Bureau employees and high school students
   - Writing styles may not represent global handwriting variations
   - Cultural differences in digit formation (e.g., the way "7" is written in Europe vs. US) are underrepresented

2. **Demographic Bias:**
   - Age bias: Primarily adult handwriting, limited children's or elderly handwriting samples
   - Educational bias: Samples from educated individuals may not represent all literacy levels
   - Geographic bias: Limited representation of non-Western handwriting styles

3. **Technical Bias:**
   - Image quality bias: All images are 28x28 grayscale, may not generalize to different resolutions or color images
   - Preprocessing bias: Centering and normalization may favor certain writing styles

#### **Potential Real-World Impact:**
- **Healthcare**: Misreading handwritten medical prescriptions could lead to medication errors
- **Education**: Automated grading systems might unfairly penalize students with different handwriting styles
- **Banking**: Check processing systems might discriminate against certain demographic groups

#### **Mitigation Strategies:**

1. **Using TensorFlow Fairness Indicators:**
```python
# Example implementation for bias detection
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators

# Define sensitive groups (if demographic data available)
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=[
        tfma.SlicingSpec(),  # Overall
        tfma.SlicingSpec(feature_keys=['age_group']),  # By age
        tfma.SlicingSpec(feature_keys=['geographic_region'])  # By region
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='FairnessIndicators',
                    config='{"thresholds": [0.1, 0.3, 0.5, 0.7, 0.9]}')
            ]
        )
    ]
)
```

2. **Data Augmentation for Fairness:**
   - Collect additional samples from underrepresented groups
   - Apply style transfer techniques to simulate different handwriting styles
   - Include synthetic data generation for minority writing patterns

3. **Model Evaluation Across Groups:**
   - Test performance across different demographic groups
   - Monitor for differential performance
   - Implement threshold adjustment for equalized odds

---

### **Amazon Reviews NLP Model**

#### **Potential Biases:**

1. **Sentiment Analysis Bias:**
   - **Gender Bias**: Language patterns associated with different genders may be classified differently
   - **Cultural Bias**: Expressions of sentiment vary across cultures and languages
   - **Socioeconomic Bias**: Vocabulary and writing styles may correlate with economic status

2. **Named Entity Recognition Bias:**
   - **Brand Bias**: Popular Western brands may be recognized more accurately than international brands
   - **Product Category Bias**: Tech products vs. traditional products recognition accuracy
   - **Language Bias**: English-centric models may perform poorly on transliterated names

3. **Representation Bias:**
   - Amazon reviews may overrepresent certain demographics (tech-savvy, higher income)
   - Product categories may have skewed review patterns
   - Geographic bias toward regions with high Amazon usage

#### **Potential Real-World Impact:**
- **Market Research**: Biased sentiment analysis could lead to poor business decisions
- **Product Recommendations**: Unfair treatment of products from certain regions or categories
- **Consumer Rights**: Automated review filtering might silence legitimate concerns from certain groups

#### **Mitigation Strategies:**

1. **Using spaCy's Rule-Based Systems for Bias Reduction:**
```python
import spacy
from spacy.matcher import Matcher

# Create bias detection patterns
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns for potentially biased language
bias_patterns = [
    [{"LOWER": {"IN": ["cheap", "expensive"]}}, {"LOWER": {"IN": ["chinese", "indian", "american"]}}],
    [{"LOWER": {"IN": ["good", "bad"]}}, {"LOWER": "for"}, {"LOWER": {"IN": ["women", "men"]}}]
]

for pattern in bias_patterns:
    matcher.add("POTENTIAL_BIAS", [pattern])

def detect_bias_indicators(text):
    doc = nlp(text)
    matches = matcher(doc)
    return [(doc[start:end].text, "potential_bias") for match_id, start, end in matches]
```

2. **Fairness-Aware Sentiment Analysis:**
   - Implement demographic parity constraints
   - Use adversarial debiasing techniques
   - Regular bias audits with diverse test sets

3. **Inclusive Model Development:**
   - Diverse training data collection
   - Multi-cultural validation sets
   - Community-based model evaluation

---

## 2. Troubleshooting Challenge: Buggy TensorFlow Code

### **Original Buggy Code:**

```python
# BUGGY CODE - DO NOT RUN
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build model with ERRORS
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # Error 1: Wrong input shape
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')  # Error 2: Wrong activation for multiclass
])

# Compile with ERRORS
model.compile(
    optimizer='sgd',  # Error 3: Not ideal optimizer
    loss='binary_crossentropy',  # Error 4: Wrong loss function
    metrics=['accuracy']
)

# Train with ERRORS
model.fit(x_train, y_train, epochs=5, batch_size=32)  # Error 5: No data preprocessing

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```

### **Issues Identified:**

1. **Dimension Mismatch**: Dense layer expects flattened input, but MNIST data is 28x28
2. **Wrong Activation**: Using sigmoid instead of softmax for multiclass classification
3. **Suboptimal Optimizer**: SGD is slower than Adam for this task
4. **Wrong Loss Function**: Binary crossentropy for multiclass problem
5. **Missing Preprocessing**: No normalization of pixel values
6. **Missing Data Reshape**: MNIST needs to be flattened for Dense layers
7. **Missing Label Encoding**: Labels should be one-hot encoded

### **Corrected Code:**

```python
# CORRECTED CODE
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_corrected_mnist_model():
    """
    Corrected MNIST classification model with proper preprocessing and architecture.
    """
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # FIX 1: Proper data preprocessing
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # FIX 2: Reshape data for Dense layers (flatten 28x28 to 784)
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    
    # FIX 3: Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # FIX 4: Correct model architecture
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # Correct input shape
        keras.layers.Dropout(0.2),  # Add dropout for regularization
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')  # Correct activation for multiclass
    ])
    
    # FIX 5: Correct compilation
    model.compile(
        optimizer='adam',  # Better optimizer
        loss='categorical_crossentropy',  # Correct loss function
        metrics=['accuracy']
    )
    
    # Display model summary
    print("Corrected Model Architecture:")
    model.summary()
    
    # FIX 6: Proper training with validation
    history = model.fit(
        x_train, y_train,
        batch_size=128,  # Larger batch size for efficiency
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate the corrected model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\\nCorrected Model Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history

# Alternative: CNN version for better performance
def create_cnn_alternative():
    """
    CNN alternative that works better with image data.
    """
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape for CNN
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # CNN model (better for images)
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN Model Architecture:")
    model.summary()
    
    return model

if __name__ == "__main__":
    print("DEBUGGING TENSORFLOW CODE")
    print("=" * 50)
    
    # Run corrected version
    corrected_model, history = create_corrected_mnist_model()
    
    # Show improvement
    print("\\n" + "=" * 50)
    print("FIXES APPLIED:")
    print("=" * 50)
    print("✅ Fixed input shape mismatch")
    print("✅ Changed sigmoid to softmax activation")
    print("✅ Switched from SGD to Adam optimizer")
    print("✅ Changed to categorical_crossentropy loss")
    print("✅ Added proper data preprocessing")
    print("✅ Added data normalization")
    print("✅ Added one-hot encoding for labels")
    print("✅ Added dropout for regularization")
    print("✅ Added validation during training")
```

### **Debugging Process Summary:**

1. **Error Analysis**: Systematically identified each issue
2. **Dimension Debugging**: Checked tensor shapes at each layer
3. **Loss Function Validation**: Verified compatibility with problem type
4. **Performance Optimization**: Improved optimizer and added regularization
5. **Best Practices**: Added validation, proper preprocessing, and documentation

---

## 3. Bias Mitigation Framework

### **Comprehensive Bias Assessment Protocol:**

```python
class AIBiasAssessment:
    """
    A framework for assessing and mitigating bias in AI models.
    """
    
    def __init__(self, model, data, sensitive_attributes):
        self.model = model
        self.data = data
        self.sensitive_attributes = sensitive_attributes
    
    def assess_demographic_parity(self, predictions, groups):
        """Assess if positive prediction rates are equal across groups."""
        parity_scores = {}
        for group in groups:
            group_data = predictions[groups == group]
            parity_scores[group] = np.mean(group_data)
        return parity_scores
    
    def assess_equalized_odds(self, predictions, true_labels, groups):
        """Assess if TPR and FPR are equal across groups."""
        from sklearn.metrics import confusion_matrix
        
        metrics = {}
        for group in np.unique(groups):
            group_mask = groups == group
            group_pred = predictions[group_mask]
            group_true = true_labels[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics[group] = {'TPR': tpr, 'FPR': fpr}
        
        return metrics
    
    def generate_bias_report(self):
        """Generate comprehensive bias assessment report."""
        report = {
            'demographic_parity': self.assess_demographic_parity(),
            'equalized_odds': self.assess_equalized_odds(),
            'recommendations': self.get_mitigation_recommendations()
        }
        return report
```

---

## 4. Recommendations for Ethical AI Development

### **For Machine Learning Projects:**

1. **Pre-Development:**
   - Conduct bias impact assessments
   - Ensure diverse development teams
   - Plan for inclusive data collection

2. **During Development:**
   - Regular bias testing throughout development
   - Implement fairness constraints in model training
   - Use interpretable models when possible

3. **Post-Deployment:**
   - Continuous monitoring for bias drift
   - Regular model audits
   - Feedback mechanisms for affected communities

### **Tools and Resources:**

1. **TensorFlow Fairness Indicators**: Comprehensive bias detection
2. **AI Fairness 360 (IBM)**: Open-source toolkit for bias detection and mitigation
3. **What-If Tool**: Interactive visual interface for model understanding
4. **Fairlearn**: Python toolkit for fairness assessment and mitigation

### **Conclusion:**

Ethical AI development requires proactive identification and mitigation of bias throughout the entire machine learning lifecycle. By implementing systematic bias assessment protocols, using appropriate tools, and maintaining ongoing vigilance, we can develop AI systems that are fair, inclusive, and beneficial for all users.
