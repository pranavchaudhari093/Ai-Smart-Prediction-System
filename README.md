# 🤖 AI Smart Prediction System

A comprehensive, production-ready web application that leverages machine learning to provide intelligent predictions across multiple domains. Built with Flask and modern web technologies, this system offers real-time analytics, interactive visualizations, and automated reporting capabilities.

![AI Smart Prediction System](https://img.shields.io/badge/Flask-2.0+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Machine Learning Models](#-machine-learning-models)
- [Quick Tools & Utilities](#-quick-tools--utilities)
- [Project Structure](#-project-structure)
- [Database Schema](#-database-schema)
- [API Endpoints](#-api-endpoints)
- [Screenshots](#-screenshots)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### 🔐 Authentication & User Management
- **Secure User Registration** with password hashing (Werkzeug Security)
- **Login System** with session management
- **Profile Management** with photo upload capability
- **Cache Control** to prevent unauthorized access

### 🎯 Core Prediction Models

#### 1. 📚 Student Performance Prediction
- Predicts student exam scores based on:
  - Study hours per day
  - Attendance percentage
  - Previous academic marks
- **Additional Insights**:
  - Grade calculation (A/B/C/D)
  - Pass/Fail status
  - Performance labels (Excellent/Good/Average)
  - Improvement tracking
  - Personalized suggestions

#### 2. 🏠 Real Estate Price Prediction
- Mumbai house price prediction using:
  - Number of BHK rooms
  - Area (square feet)
  - Location (encoded features)
- **Market Analytics**:
  - Price range estimation (±10%)
  - Price per square foot
  - Location rating (1-5 stars)
  - Market status (Fair Deal/Average/Overpriced)
  - 5-year future value projection (8% growth)
  - Auto-formatting (Lakhs/Crores)

#### 3. ❤️ Heart Disease Risk Assessment
- Medical diagnostic tool analyzing:
  - Age, Gender, Cholesterol levels
  - Blood pressure, Heart rate
  - Lifestyle factors (smoking, alcohol, exercise)
  - Medical history (diabetes, obesity, family history)
  - Stress levels, Blood sugar
  - Chest pain type, Exercise-induced angina
- **Health Metrics**:
  - 4-level risk categorization (Low/Moderate/High/Critical)
  - Health score (0-100)
  - Major risk factor identification
  - Personalized health recommendations

#### 4. 💰 Loan Approval Prediction
- Financial credit analysis using:
  - Applicant income and employment
  - Credit history and score
  - Loan amount and interest rate
  - Debt-to-income ratio
  - Previous loan defaults
  - Education, Home ownership status
- **Output**:
  - Approval probability (%)
  - Model accuracy metrics
  - Decision recommendation (Approved/Rejected)

#### 5. 📧 Spam Email Detection
- NLP-powered email classification:
  - Text vectorization (TF-IDF)
  - Binary classification (Spam/Not Spam)
  - Real-time text analysis

#### 6. 📰 Fake News Detection
- News authenticity verification:
  - Advanced NLP processing
  - Probability scoring (0-100%)
  - Confidence threshold (≥65% = Real, <65% = Fake)
  - Content credibility assessment

### 📊 Advanced Analytics Dashboard
- **Interactive Visualizations** with Chart.js
- Time-series prediction graphs
- Model-wise performance tracking
- Historical data analysis
- Animated UI with gradient effects
- Responsive design for all devices

### 📄 Automated Report Generation
- **PDF Reports** using FPDF library
- Date-specific prediction summaries
- Quick tools usage statistics
- Professional formatting with:
  - User information section
  - Detailed prediction tables
  - AI-generated explanations
  - Downloadable PDF format

### 🛠️ Quick Tools & Utilities

#### Sentiment Analyzer
- Analyze emotions and sentiments in text
- Real-time sentiment detection

#### Password Strength Checker
- Evaluate password security
- Strength recommendations

#### Word Counter
- Text statistics and word frequency analysis
- Character and sentence counting

#### Name Generator
- Creative name suggestions for projects/businesses
- AI-powered brainstorming

#### Text Summarizer
- Condense long texts using NLP
- Extract key information

#### Idea Generator
- Innovative idea brainstorming tool
- Creative thinking assistant

### 🤖 AI Chatbot Assistant
- Rule-based conversational AI
- Contextual responses for:
  - Model explanations
  - System feature guidance
  - Report generation help
  - Quick tools information
  - Greeting and general queries

### 🎨 Modern User Interface
- **Dark Theme** professional design
- **Sliding Sidebar Navigation** (Off-canvas style)
- Smooth CSS transitions and animations
- Responsive layout (Mobile/Tablet/Desktop)
- Google Fonts integration (Inter, Plus Jakarta Sans)
- Gradient backgrounds and glassmorphism effects
- Interactive hover states and micro-animations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer (Browser)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   HTML/CSS   │  │  JavaScript  │  │  Chart.js    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/HTTPS
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer (Flask)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Routes     │  │   Sessions   │  │  File Upload │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Auth       │  │   Middleware │  │  PDF Gen     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    ML Inference Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Scikit-learn│  │   Pickle     │  │  Encoders    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer (SQLite)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Users     │  │  Predictions │  │ Quick Tools  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome, Firefox, Edge, Safari)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ai-smart-prediction-system.git
cd ai-smart-prediction-system
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install flask
pip install scikit-learn
pip install pandas
pip install numpy
pip install werkzeug
pip install fpdf
```

### Step 4: Verify Project Structure
Ensure the following structure exists:
```
AI_Smart_Prediction_System/
├── app.py                    # Main Flask application
├── datasets/                 # Training data files
├── models/                   # Pre-trained .pkl model files
├── notebooks/               # Jupyter notebooks for model training
├── static/                  # CSS, JS, images
├── templates/               # HTML templates
├── users.db                # SQLite database (auto-created)
└── requirements.txt        # Dependencies list
```

### Step 5: Run the Application
```bash
python app.py
```

The server will start at: `http://127.0.0.1:5000/`

---

## 🚀 Usage

### First-Time Setup
1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Click on **Register** to create a new account
3. Fill in your details (Name, Email, Password)
4. After registration, **Login** with your credentials

### Making Predictions
1. From the homepage, select any AI model card
2. Fill in the required input fields
3. Click **Predict** to get instant results
4. View detailed analytics and insights

### Viewing Analytics
1. Navigate to **Analytics** from the dashboard
2. Explore interactive charts showing your prediction history
3. Filter by date range and model type

### Generating Reports
1. Go to **Reports** section
2. Select a specific date
3. Click **View Report** to see predictions for that day
4. Click **Download PDF** to save the report locally

### Using Quick Tools
1. Access quick tools from the dashboard
2. Try Sentiment Analysis, Password Checker, Word Counter, etc.
3. All tool usage is tracked in your activity log

---

## 🧠 Machine Learning Models

### Model Training Details

All models were trained using **Scikit-learn** and serialized with **pickle**.

| Model | Algorithm | Dataset | Accuracy |
|-------|-----------|---------|----------|
| Student Prediction | Linear Regression | student.csv | ~85% |
| House Price | Linear Regression | mumbai_house_price.csv | ~88% |
| Heart Disease | Logistic Regression | heart_disease_dataset.csv | ~91% |
| Loan Approval | Random Forest Classifier | loan_data.csv | ~94% |
| Spam Detection | Naive Bayes + TF-IDF | spam.csv | ~97% |
| Fake News | Logistic Regression + TF-IDF | Fake_news.csv | ~92% |

### Model Files
- `student_model.pkl` - Student performance predictor
- `house_price_mumbai.pkl` - Mumbai real estate estimator
- `heart_model.pkl` - Cardiac risk assessor
- `loan_model.pkl` - Loan approval classifier
- `spam_model.pkl` - Email spam detector
- `fake_news_model.pkl` - News authenticity verifier

### Feature Encoding
Categorical variables are encoded using:
- **Label Encoding** for ordinal features
- **One-Hot Encoding** for nominal features
- Stored in separate `.pkl` encoder files

---

## 🛠️ Quick Tools & Utilities

### 1. Sentiment Analyzer
Analyzes text emotion and polarity using NLP techniques.

### 2. Password Strength Checker
Evaluates password security based on:
- Length requirements
- Character variety (uppercase, lowercase, numbers, symbols)
- Common pattern detection

### 3. Word Counter
Provides text statistics:
- Word count, character count
- Sentence and paragraph count
- Reading time estimation
- Most frequent words

### 4. Name Generator
Generates creative names using:
- Keyword combinations
- Industry-specific patterns
- Brand naming principles

### 5. Text Summarizer
Condenses long documents using:
- Extractive summarization
- Key sentence identification
- Important phrase extraction

### 6. Idea Generator
Brainstorms innovative concepts for:
- Business startups
- Creative projects
- Problem-solving scenarios

---

## 📁 Project Structure

```
AI_Smart_Prediction_System/
│
├── app.py                          # Main Flask application (1045 lines)
│   ├── Database initialization
│   ├── Route handlers (30+ endpoints)
│   ├── ML model loading
│   ├── Prediction logic
│   ├── Report generation
│   └── Chatbot responses
│
├── datasets/                       # Raw training data (6 files)
│   ├── student.csv                 # Student performance data
│   ├── mumbai_house_price.csv      # Real estate dataset
│   ├── heart_disease_dataset.csv   # Medical records
│   ├── loan_data.csv               # Financial loan applications
│   ├── spam.csv                    # Email classification
│   └── Fake_news.csv               # News articles (Real/Fake)
│
├── models/                         # Serialized ML models (12 files)
│   ├── student_model.pkl
│   ├── house_price_mumbai.pkl
│   ├── heart_model.pkl
│   ├── heart_encoders.pkl
│   ├── loan_model.pkl
│   ├── loan_encoders.pkl
│   ├── loan_accuracy.pkl
│   ├── spam_model.pkl
│   ├── spam_vectorizer.pkl
│   ├── fake_news_model.pkl
│   ├── fake_news_vectorizer.pkl
│   └── location_encoder.pkl
│
├── notebooks/                      # Model training scripts (6 files)
│   ├── student_prediction.py
│   ├── house_price_prediction.py
│   ├── heart_disease_model.py
│   ├── loan_data_model.py
│   ├── spam_model.py
│   └── fake_news_model.py
│
├── static/                         # Frontend assets (14 files)
│   ├── home.css                    # Homepage styling
│   ├── user_dashboard.css          # Dashboard dark theme
│   ├── student.css                 # Student page styles
│   ├── house.css                   # House prediction styles
│   ├── heart.css                   # Health analysis styles
│   ├── loan.css                    # Loan page styles
│   ├── spam.css                    # Spam detector styles
│   ├── fake.css                    # Fake news styles
│   ├── login.css                   # Login page styles
│   ├── register.css                # Registration styles
│   ├── user_dashboard.css          # Modern sidebar dashboard
│   └── dashboard_toggle.js         # Sidebar toggle script
│
├── templates/                      # HTML templates (19 files)
│   ├── home.html                   # Landing page with model cards
│   ├── login.html                  # User login
│   ├── register.html               # User registration
│   ├── user_dashboard.html         # Modern dashboard with sidebar
│   ├── student.html                # Student prediction interface
│   ├── house.html                  # House price interface
│   ├── heart.html                  # Heart disease interface
│   ├── loan.html                   # Loan approval interface
│   ├── spam.html                   # Spam detection interface
│   ├── fake.html                   # Fake news interface
│   ├── analytics.html              # Chart.js visualizations
│   ├── report.html                 # PDF report viewer
│   ├── select_report_date.html     # Date picker for reports
│   ├── sentiment_analyzer.html     # Sentiment tool UI
│   ├── password_checker.html       # Password tool UI
│   ├── word_counter.html           # Word counter UI
│   ├── name_generator.html         # Name generator UI
│   ├── text_summarizer.html        # Summarizer UI
│   └── idea_generator.html         # Idea generator UI
│
├── users.db                        # SQLite database (auto-generated)
├── .hintrc                         # Linter configuration
└── README.md                       # This file
```

---

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL  -- Hashed with Werkzeug
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    prediction_value REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Quick Tools Usage Table
```sql
CREATE TABLE quick_tools_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/login` | Login page |
| POST | `/login` | Authenticate user |
| GET | `/register` | Registration page |
| POST | `/register` | Create new account |
| GET | `/logout` | Clear session |

### Prediction Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/student` | Student prediction page |
| POST | `/predict` | Generate student score |
| GET | `/house` | House price page |
| POST | `/house_predict` | Calculate property value |
| GET | `/heart` | Heart disease page |
| POST | `/heart_predict` | Assess cardiac risk |
| GET | `/loan` | Loan approval page |
| POST | `/loan_predict` | Predict loan decision |
| GET | `/spam` | Spam detector page |
| POST | `/spam_predict` | Classify email |
| GET | `/fake` | Fake news page |
| POST | `/fake_predict` | Verify news authenticity |

### Dashboard & Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/user_dashboard` | User dashboard with sidebar |
| GET | `/analytics` | Interactive charts |
| GET | `/select_report_date` | Report date selection |
| POST | `/view_report` | Display daily predictions |
| POST | `/generate_report` | Download PDF report |

### Quick Tools
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sentiment` | Sentiment analyzer |
| GET | `/password` | Password checker |
| GET | `/word_counter` | Word statistics |
| GET | `/name_generator` | Creative name generator |
| GET | `/summarizer` | Text summarizer |
| GET | `/idea` | Idea brainstorming |

### AI Assistant
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chatbot` | Chat interface |
| POST | `/chatbot_response` | Get AI replies |

### Profile Management
| POST | `/upload_profile` | Upload profile photo |

---

## 📸 Screenshots

*(Add your screenshots here)*

### Homepage
![Homepage](screenshots/home.png)

### User Dashboard
![Dashboard](screenshots/dashboard.png)

### Analytics
![Analytics](screenshots/analytics.png)

### Report Generation
![Report](screenshots/report.png)

---

## 🛠️ Technologies Used

### Backend
- **Flask** (v2.0+) - Web framework
- **SQLite** - Database
- **Scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Werkzeug** - Security utilities
- **FPDF** - PDF generation
- **Pickle** - Model serialization

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Styling with gradients, animations
- **JavaScript (ES6+)** - Interactivity
- **Chart.js** - Data visualization
- **Google Fonts** - Typography (Inter, Plus Jakarta Sans)
- **Font Awesome** - Icons
- **Tailwind CSS** (via CDN) - Utility classes

### Development Tools
- **Visual Studio Code** - Code editor
- **Jupyter Notebooks** - Model experimentation
- **Git** - Version control

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Write clean, documented code
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation accordingly
- Respect the existing codebase structure

---

## 📄 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

**Summary**: You are free to use, modify, distribute, and sell this software. The authors are not liable for any damages or claims.

---

## 📞 Contact

### Project Maintainer
- **Name**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [linkedin.com/in/yourprofile]
- **GitHub**: [github.com/yourusername]

### Acknowledgments
- Machine learning models trained on publicly available datasets
- UI inspired by modern dashboard designs
- Built with passion for AI and education

---

## 🙏 Support

If you find this project helpful, please consider:
- ⭐ Starring this repository
- 🍴 Forking and contributing
- 📢 Sharing with others
- 💡 Providing feedback and suggestions

---

## 🚀 Future Enhancements

- [ ] Deploy to cloud (AWS/GCP/Heroku)
- [ ] Add REST API support
- [ ] Integrate deep learning models (TensorFlow/PyTorch)
- [ ] Implement user role management (Admin/User)
- [ ] Add email notifications
- [ ] Create mobile app version
- [ ] Support multiple languages
- [ ] Add data export options (CSV, Excel)
- [ ] Implement model retraining pipeline
- [ ] Add A/B testing framework

---

## 📊 Project Statistics

- **Total Lines of Code**: ~5,000+
- **Backend Code**: 1,045 lines (app.py)
- **Frontend Templates**: 19 HTML files
- **ML Models**: 6 production-ready models
- **Quick Tools**: 6 utility applications
- **Database Tables**: 3 relational tables
- **API Endpoints**: 30+ routes

---

**Made with ❤️ and ☕ by the AI Development Team**

*Last Updated: February 2026*
