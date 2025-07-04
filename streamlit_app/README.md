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
- **🌸 Task 1**: Interactive Iris classification with Decision Trees
- **🔢 Task 2**: MNIST CNN results and model architecture
- **📝 Task 3**: NLP analysis of Amazon product reviews

### Interactive Testing
- **🌸 Iris Predictor**: Real-time species prediction with input sliders
- **🔢 Digit Classifier**: Draw digits or upload images for classification
- **📝 Review Analyzer**: Live sentiment analysis and brand extraction

### Features by Page

#### Home Dashboard
- Performance metrics for all three tasks
- Technology stack overview
- Project summary and achievements

#### Task 1: Iris Classification
- Real-time model training and evaluation
- Interactive confusion matrix
- Feature importance visualization
- Classification performance metrics

#### Task 2: MNIST CNN
- Model architecture display
- Per-class performance analysis
- Training configuration details
- Key insights and achievements

#### Task 3: NLP Reviews
- Sample review analysis
- Brand mention frequency
- Sentiment distribution charts
- Product feature sentiment analysis

#### Interactive Prediction Pages

##### Iris Predictor
- Real-time species prediction using slider inputs
- Probability distribution visualization
- Species information and characteristics
- Input summary table

##### Digit Classifier
- Simulated drawing canvas for digit input
- Image upload functionality for digit classification
- Confidence scores for all digit classes
- Model architecture information

##### Review Analyzer
- Text input area for product reviews
- Sample review selection
- Real-time sentiment analysis with scoring
- Brand and entity extraction
- Word analysis and insights

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
