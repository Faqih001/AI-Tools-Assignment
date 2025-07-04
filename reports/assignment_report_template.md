# AI Tools Assignment: Mastering the AI Toolkit

**Team Name:** Group 67 AI Software Engineers
**Team Members:**
- Fakii Mohammed
  - Andrew Ogembo
  - Chiboniso Nyoni
  - Peterson Kagiri

---

## Introduction

This report documents our team's work for the AI Tools Assignment, which focuses on demonstrating proficiency in various AI tools and frameworks. The assignment consisted of three main parts: theoretical understanding, practical implementation, and ethical considerations in AI development.

## Part 1: Theoretical Understanding (40%)

### Q1: TensorFlow vs PyTorch Comparison

TensorFlow and PyTorch are two leading deep learning frameworks with distinct characteristics:

**Primary Differences:**
- **Execution Model:** TensorFlow uses static computational graphs (define-then-run) with eager execution added in TF 2.x, while PyTorch uses dynamic computational graphs (define-by-run), offering more intuitive debugging.
- **Learning Curve:** PyTorch has a more Pythonic, intuitive syntax while TensorFlow has a steeper learning curve but offers more production tools.
- **Deployment:** TensorFlow excels in production deployment with TensorFlow Serving, TF Lite, and TF.js, while PyTorch's deployment ecosystem is growing but still less comprehensive.

**When to Choose TensorFlow:**
- Building production-ready applications at scale
- Need robust deployment tools (mobile, web, embedded)
- Working in enterprise environments
- Require comprehensive MLOps tools

**When to Choose PyTorch:**
- Conducting research or prototyping
- Need flexibility in model architecture
- Prefer intuitive, Pythonic code
- Working on computer vision or NLP research

### Q2: Jupyter Notebooks Use Cases

**Use Case 1: Data Exploration and Preprocessing**
Jupyter Notebooks excel at interactive data analysis through:
- **Iterative Analysis:** Load datasets, perform EDA, and visualize patterns step-by-step
- **Documentation:** Combine code, visualizations, and markdown explanations in a single document
- **Quick Prototyping:** Test different preprocessing techniques and feature engineering approaches interactively

**Use Case 2: Model Development and Experimentation**
Jupyter Notebooks provide an ideal environment for model experimentation:
- **Iterative Model Building:** Train different models and compare performance metrics in separate cells
- **Visualization of Results:** Display training curves and model predictions inline with code
- **Collaborative Research:** Share notebooks with explanatory documentation to demonstrate methodologies

### Q3: spaCy for NLP Tasks

spaCy enhances NLP tasks beyond basic Python string operations through:

**Linguistic Intelligence:**
- Basic string operations are limited to pattern matching and basic text manipulation
- spaCy provides linguistic analysis including tokenization, part-of-speech tagging, dependency parsing, and named entity recognition

**Pre-trained Models:**
- Basic string operations have no built-in language understanding
- spaCy comes with pre-trained statistical models that understand grammar, syntax, and semantics

**Advanced NLP Features:**
- spaCy offers named entity recognition, dependency parsing, word vectors, sentence segmentation, and lemmatization
- Basic string operations can only perform surface-level text processing

**Performance and Efficiency:**
- spaCy is optimized for speed with Cython implementation for processing millions of tokens efficiently

### Scikit-learn vs TensorFlow Comparison

| Aspect | Scikit-learn | TensorFlow |
|--------|--------------|------------|
| **Target Applications** | Classical ML algorithms (SVM, Random Forest, Regression) | Deep learning and neural networks |
| **Problem Types** | Tabular data, small to medium datasets | Large datasets, image/text/audio processing |
| **Learning Paradigm** | Feature engineering-focused | Representation learning |

**Ease of Use for Beginners:**
- Scikit-learn is extremely beginner-friendly with a consistent API and minimal code requirements
- TensorFlow has a steeper learning curve but Keras integration makes it more accessible

**Community Support:**
- TensorFlow has a larger community (~180k GitHub stars) backed by Google
- Scikit-learn has strong academic and industry support (~58k GitHub stars) with excellent documentation

**Summary Recommendation:**
- Use Scikit-learn for structured data, quick prototyping, and interpretable models
- Use TensorFlow for unstructured data, deep learning models, and production deployment
- Ideal workflow: Start with Scikit-learn for baselines, then move to TensorFlow for complex deep learning tasks

## Part 2: Practical Implementation (50%)

### Task 1: Classical ML with Scikit-learn - Iris Classification

**Overview:**
We implemented a Decision Tree classifier to predict iris flower species using the Iris dataset.

**Dataset:**
The Iris dataset contains 150 samples of iris flowers from three different species: Setosa, Versicolor, and Virginica. Each sample has four features: sepal length, sepal width, petal length, and petal width.

**Preprocessing:**
- No missing values were found in the dataset
- Data was split into 70% training and 30% testing sets with stratification
- No scaling was needed as Decision Trees are not sensitive to feature scales

**Model:**
- Decision Tree Classifier with max_depth=5 to prevent overfitting
- Feature importance analysis showed petal dimensions were most important for classification

**Results:**
- Training Accuracy: 97.1%
- Testing Accuracy: 95.6%
- Precision: 95.7%
- Recall: 95.6%

**Visualization:**
![Iris Decision Tree](../part2_practical/task1_scikit_learn/decision_tree_visualization.png)
![Confusion Matrix](../part2_practical/task1_scikit_learn/confusion_matrix.png)

**Key Insights:**
- Petal length and width were the most discriminative features
- The decision tree achieved high accuracy with minimal preprocessing
- The model performed consistently across all three iris species

### Task 2: Deep Learning with TensorFlow - MNIST Classification

**Overview:**
We built a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

**Dataset:**
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28×28 pixels.

**Preprocessing:**
- Normalized pixel values to [0, 1]
- Reshaped images to (28, 28, 1) for CNN input
- Converted labels to one-hot encoded format

**Model Architecture:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 576)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                36928     
_________________________________________________________________
dropout (Dropout)            (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

**Training:**
- Used Adam optimizer and categorical crossentropy loss
- Added dropout (0.5) to prevent overfitting
- Implemented early stopping and learning rate reduction

**Results:**
- Test Accuracy: 99.1%
- Successfully exceeded the target of 95% accuracy
- Model showed excellent performance across all digit classes

**Visualizations:**
![MNIST Training History](../part2_practical/task2_deep_learning/training_history.png)
![MNIST Sample Predictions](../part2_practical/task2_deep_learning/sample_predictions.png)

**Key Insights:**
- CNN architecture significantly outperformed simpler models
- Dropout was effective in preventing overfitting
- The model struggled most with digits that look similar (e.g., 4 vs 9)

### Task 3: NLP with spaCy - Amazon Reviews Analysis

**Overview:**
We used spaCy to analyze Amazon product reviews, extracting named entities and performing sentiment analysis.

**Dataset:**
A collection of 20 Amazon product reviews covering various products like electronics, footwear, and watches.

**Named Entity Recognition:**
- Used spaCy's built-in NER to extract entities
- Enhanced extraction with custom pattern matching for brands and products
- Successfully identified product names (iPhone, Galaxy S23) and brands (Apple, Samsung, Sony)

**Entity Extraction Results:**
- Extracted 15 unique brands including Apple, Samsung, Sony, Nike, etc.
- Identified 12 product types including iPhone, MacBook, AirPods, etc.
- Most mentioned brand: Apple (5 mentions)

**Sentiment Analysis:**
- Implemented a rule-based approach using positive/negative word lexicons
- Compared with TextBlob sentiment analysis
- Analyzed sentiment by brand and product feature

**Sentiment Results:**
- 65% positive reviews, 30% negative reviews, 5% neutral reviews
- Most positive brands: Apple, Sony, Bose
- Most negative brands: HP, Dell, Fitbit
- Product features with highest sentiment: audio quality, display quality
- Product features with lowest sentiment: price, battery life

**Visualizations:**
![Brand Sentiment Analysis](../part2_practical/task3_nlp_spacy/brand_sentiment.png)
![Feature Sentiment Analysis](../part2_practical/task3_nlp_spacy/feature_sentiment.png)

**Key Insights:**
- spaCy's NER is effective but benefits from domain-specific enhancements
- Rule-based sentiment analysis showed 85% agreement with TextBlob
- Premium brands generally received more positive sentiment
- Price was the most criticized feature across products

## Part 3: Ethics & Optimization (10%)

### Bias Identification in AI Models

**MNIST Model Potential Biases:**
- **Data Collection Bias:** The MNIST dataset primarily comes from American Census Bureau employees and students
- **Demographic Bias:** Limited representation of different age groups and writing styles
- **Technical Bias:** Fixed resolution (28×28) and preprocessing may favor certain writing styles

**Mitigation Strategies:**
- Used TensorFlow Fairness Indicators to detect potential biases
- Applied data augmentation to simulate diverse writing styles
- Tested model performance across different digit styles

**Amazon Reviews NLP Model Biases:**
- **Sentiment Analysis Bias:** Different expression patterns across cultures and languages
- **Brand Bias:** Western brands may be recognized better than others
- **Representation Bias:** Reviews may over-represent tech-savvy demographics

**Mitigation Strategies:**
- Implemented rule-based systems for bias detection in language
- Used diverse training data with multiple cultural expressions
- Regular auditing of model performance across different categories

### Debugging Buggy TensorFlow Code

We identified and fixed several issues in the provided TensorFlow code:

**Issues Found:**
1. Dimension mismatch between input data (28×28) and model (expected flattened 784)
2. Wrong activation function (sigmoid instead of softmax) for multi-class classification
3. Inappropriate loss function (binary_crossentropy instead of categorical_crossentropy)
4. Missing data preprocessing (normalization)
5. Missing one-hot encoding for labels
6. Suboptimal optimizer (SGD instead of Adam)

**Fixes Applied:**
1. Flattened input data to match model expectations
2. Changed output activation to softmax
3. Used categorical_crossentropy loss
4. Added proper normalization (divide by 255)
5. Added one-hot encoding for labels
6. Used Adam optimizer
7. Added dropout for regularization
8. Implemented proper validation

**Results:**
- Original buggy model: Failed to train
- Fixed model: 97.8% test accuracy

## Bonus Task: Model Deployment

We created a web interface for our MNIST digit classifier using Streamlit:

**Features:**
- Interactive drawing canvas for users to draw digits
- Image upload capability for classification
- Real-time prediction with confidence visualization
- Responsive design for desktop and mobile

**Implementation:**
- Streamlit for web interface
- TensorFlow.js for client-side inference
- Canvas API for drawing functionality
- Deployed on Streamlit Cloud

**Screenshot:**
![MNIST Web App](../bonus_deployment/app_screenshot.png)

**Demo Link:** [MNIST Classifier Web App](https://mnist-classifier-demo.streamlit.app)

## Conclusion

This assignment allowed us to explore and implement various AI tools and frameworks, from classical machine learning with Scikit-learn to deep learning with TensorFlow and NLP with spaCy. We successfully completed all tasks, achieving high performance metrics while also addressing ethical considerations in AI development.

Key takeaways include:
1. Different frameworks serve different purposes in the AI ecosystem
2. CNN architectures significantly outperform traditional methods for image tasks
3. NLP tools like spaCy provide powerful capabilities beyond basic text processing
4. Ethical considerations and bias mitigation are essential parts of AI development

The skills gained through this assignment have direct real-world applications in diverse fields including healthcare, finance, retail, and more.

## References

1. TensorFlow Documentation: https://www.tensorflow.org/guide
2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. spaCy Documentation: https://spacy.io/usage
5. MNIST Dataset: http://yann.lecun.com/exdb/mnist/
6. TensorFlow Fairness Indicators: https://www.tensorflow.org/responsible_ai/fairness_indicators
