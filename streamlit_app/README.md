# AI Tools Assignment - Streamlit Dashboard

A comprehensive web application showcasing three AI tasks: Classical ML, Deep Learning, and NLP.

## Features

### 🔐 Authentication System
- User registration and login
- Secure password hashing
- Session management
- User profile information

### 📊 Dashboard Pages
1. **Home Dashboard** - Overview of all three tasks
2. **Task 1: Iris Classification** - Classical ML with Decision Trees
3. **Task 2: MNIST CNN** - Deep Learning image classification
4. **Task 3: NLP Reviews** - Natural Language Processing analysis

### 🧪 Interactive Prediction Pages
1. **Iris Predictor** - Live species prediction with sliders
2. **Digit Classifier** - Draw or upload digits for classification
3. **Review Analyzer** - Real-time sentiment and entity analysis

### 🎨 UI Components
- Professional header with gradient design
- Responsive sidebar navigation
- Metrics cards and visualizations
- Interactive charts and graphs
- Clean, modern interface

## Installation & Setup

### 1. Create Virtual Environment
```bash
cd streamlit_app
python -m venv streamlit_env
source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy Model (for NLP tasks)
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the Application

#### Option 1: Using the Quick Start Script
```bash
chmod +x run_app.sh
./run_app.sh
```

#### Option 2: Manual Command
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### First Time Setup
1. **Register Account**: Create a new user account with username, email, and password
2. **Login**: Use your credentials to access the dashboard
3. **Navigate**: Use the sidebar to explore different AI tasks

### Navigation
- **🏠 Home**: Dashboard overview with metrics and project summary

#### 📋 Navigate to: (Task Summary Pages)
- **🌸 Task 1**: Iris Classification results and model performance summary
- **🔢 Task 2**: MNIST CNN results, architecture, and training metrics
- **📝 Task 3**: NLP analysis results and sample review insights

#### 🧪 Test Predictions: (Interactive Testing Pages)
- **🌸 Iris Predictor**: Real-time species prediction with input sliders
- **🔢 Digit Classifier**: Draw digits with mouse or upload images for classification
- **📝 Review Analyzer**: Live sentiment analysis and brand extraction

### Features by Page

#### Home Dashboard
- Performance metrics for all three tasks
- Technology stack overview
- Project summary and achievements

#### Task Summary Pages (📋 Navigate to:)

##### Task 1: Iris Classification Summary
- Model training results and evaluation metrics
- Interactive confusion matrix and performance analysis
- Feature importance visualization
- Classification accuracy and detailed statistics

##### Task 2: MNIST CNN Summary
- Model architecture display and layer details
- Per-class performance analysis and confusion matrix
- Training configuration and hyperparameters
- Key insights and achievements (99.31% accuracy)

##### Task 3: NLP Reviews Summary
- Sample review analysis and processing results
- Brand mention frequency and extraction patterns
- Sentiment distribution charts and statistics
- Product feature sentiment analysis insights

#### Interactive Testing Pages (🧪 Test Predictions:)

##### Iris Predictor
- Real-time species prediction using slider inputs
- Probability distribution visualization for all species
- Species information and characteristics
- Input summary table with current measurements

##### Digit Classifier
- Mouse-based drawing canvas for digit input (black background, white brush)
- Real-time prediction as you draw
- Image upload functionality for digit classification
- Confidence scores visualization for all digit classes (0-9)
- Brush size and canvas controls
- Technical details and drawing statistics

##### Review Analyzer
- Text input area for custom product reviews
- Sample review selection for quick testing
- Real-time sentiment analysis with confidence scoring
- Brand and entity extraction with highlighting
- Word analysis and linguistic insights

## Security Features

- **Password Hashing**: SHA-256 encryption for user passwords
- **Session Management**: Secure login state management
- **User Data**: Local JSON storage for user accounts
- **Input Validation**: Form validation and error handling

## Technical Stack

### Backend
- **Streamlit**: Web application framework
- **Python**: Core programming language
- **JSON**: User data storage

### Machine Learning
- **scikit-learn**: Classical ML algorithms
- **TensorFlow/Keras**: Deep learning models
- **spaCy**: Natural language processing
- **TextBlob**: Sentiment analysis

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts
- **Streamlit Charts**: Built-in chart components

## File Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── run_app.sh         # Quick start script
├── README.md          # This file
├── users.json         # User accounts (auto-generated)
└── .streamlit/        # Streamlit configuration (optional)
```

## Customization

### Adding New Tasks
1. Create a new page function (e.g., `task4_new_page()`)
2. Add navigation option in `sidebar_navigation()`
3. Add routing in `main()` function

### Modifying Styling
- Update CSS in `show_header()` function
- Modify color schemes in chart functions
- Customize layout using Streamlit columns

### Database Integration
- Replace JSON user storage with SQLite/PostgreSQL
- Add user roles and permissions
- Implement advanced authentication features

## Deployment Options

### Local Development

#### Quick Start (Recommended)
```bash
chmod +x run_app.sh
./run_app.sh
```

#### Manual Start
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with automatic requirements detection

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **TensorFlow Installation Issues**
   ```bash
   pip install --upgrade tensorflow
   ```

3. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Memory Issues with Large Models**
   - Reduce batch sizes in model demonstrations
   - Use model summaries instead of full model loading

### Performance Optimization
- Cache expensive computations with `@st.cache_data`
- Use `@st.cache_resource` for model loading
- Optimize image loading and processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of an AI Tools Assignment and is intended for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Streamlit documentation
3. Contact the development team

---

**Note**: This application demonstrates AI capabilities across multiple domains and serves as a comprehensive portfolio piece showcasing machine learning, deep learning, and natural language processing skills.
