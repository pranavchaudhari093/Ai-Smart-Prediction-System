from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pickle
from functools import wraps
import os
from flask import render_template, request, send_file
import sqlite3
from fpdf import FPDF
import datetime
import io
app = Flask(__name__)
app.secret_key = "super_secret_key"

# ================= FILE UPLOAD CONFIG =================
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= PREDICTION HISTORY TABLE =================
from datetime import datetime

def init_prediction_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_name TEXT,
            prediction_value REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_prediction_table()

def save_prediction(user_id, model_name, value):
    conn = get_db()
    conn.execute("""
        INSERT INTO predictions (user_id, model_name, prediction_value, created_at)
        VALUES (?, ?, ?, ?)
    """, (user_id, model_name, value, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ================= QUICK TOOLS USAGE TABLE =================
def init_quick_tools_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quick_tools_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tool_name TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_quick_tools_table()

def save_quick_tool_usage(user_id, tool_name):
    conn = get_db()
    conn.execute("""
        INSERT INTO quick_tools_usage (user_id, tool_name, created_at)
        VALUES (?, ?, ?)
    """, (user_id, tool_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ================= CACHE CONTROL =================
@app.after_request
def add_cache_headers(response):
    """Prevent caching of user-specific pages"""
    if request.path in ['/', '/user_dashboard', '/analytics']:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# ================= LOGIN REQUIRED =================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

# ================= LOAD MODELS =================

def _safe_load(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"⚠️ Could not load model {path}: {e}")
        return None

student_model = _safe_load("models/student_model.pkl")

house_model = _safe_load("models/house_price_mumbai.pkl")
location_encoder = _safe_load("models/location_encoder.pkl")

heart_model = _safe_load("models/heart_model.pkl")
heart_encoders = _safe_load("models/heart_encoders.pkl")

spam_model = _safe_load("models/spam_model.pkl")
spam_vectorizer = _safe_load("models/spam_vectorizer.pkl")

# Fake News Model - Ensemble approach
fake_news_data = _safe_load("models/fake_news_model.pkl")
if fake_news_data and isinstance(fake_news_data, dict):
    fake_lr_model = fake_news_data.get('lr_model')
    fake_nb_model = fake_news_data.get('nb_model')
    fake_accuracy = fake_news_data.get('accuracy', 0)
else:
    fake_lr_model = fake_news_data
    fake_nb_model = None
    fake_accuracy = 0
    
fake_vectorizer = _safe_load("models/fake_news_vectorizer.pkl")

# Loan Model - Ensemble approach
loan_data = _safe_load("models/loan_model.pkl")
if loan_data and isinstance(loan_data, dict):
    # New ensemble model
    loan_lr_model = loan_data.get('lr_model')
    loan_rf_model = loan_data.get('rf_model')
    loan_gb_model = loan_data.get('gb_model')
    loan_weights = loan_data.get('weights', [0.3, 0.35, 0.35])
    loan_encoders = loan_data.get('encoders')
    loan_scaler = loan_data.get('scaler')
    loan_accuracy = loan_data.get('accuracy', 0)
else:
    # Fallback to old single model
    loan_lr_model = loan_data
    loan_rf_model = None
    loan_gb_model = None
    loan_weights = [1.0]
    loan_encoders = _safe_load("models/loan_encoders.pkl")
    loan_scaler = None
    loan_accuracy = _safe_load("models/loan_accuracy.pkl") or 0

# ================= HOME =================
@app.route("/")
def home():
    return render_template("home.html", user=session.get("user"))

# ================= AUTH =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "photo": None
            }
            return redirect("/")

        return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (name,email,password) VALUES (?,?,?)",
                (
                    request.form["name"],
                    request.form["email"],
                    generate_password_hash(request.form["password"])
                )
            )
            conn.commit()
            conn.close()
            return redirect("/login")
        except:
            return render_template("register.html", error="Email already exists")

    return render_template("register.html")

@app.route("/logout") 
def logout():
    session.clear()
    response = redirect("/")
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ================= AI REPORTING =================

@app.route('/select_report_date')
@login_required
def select_report_date():
    return render_template("select_report_date.html")

@app.route('/view_report', methods=['POST'])
@login_required
def view_report():

    selected_date = request.form['date']

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT model_name, prediction_value, created_at
    FROM predictions
    WHERE user_id = ? AND DATE(created_at) = ?
    """, (session["user"]["id"], selected_date))

    data = cursor.fetchall()

    # Fetch quick tools usage
    cursor.execute("""
    SELECT tool_name, created_at
    FROM quick_tools_usage
    WHERE user_id = ? AND DATE(created_at) = ?
    """, (session["user"]["id"], selected_date))

    quick_tools_data = cursor.fetchall()
    conn.close()

    # ===== Format prediction values =====
    def format_value(model_name, value):
        if "House" in model_name:
            # Format price as Cr or Lakh
            val = float(value)
            if val >= 100:
                return f"₹{val/100:.2f} Cr"
            else:
                return f"₹{val:.2f} Lakh"
        elif "Loan" in model_name:
            # Show as percentage
            return f"{float(value):.2f}%"
        elif "Fake" in model_name or "Spam" in model_name:
            # Show as percentage
            return f"{float(value):.2f}%"
        else:
            # Show as-is
            return f"{float(value):.2f}"

    # Keep raw data for chart and format for display
    formatted_data = [(row[0], format_value(row[0], row[1]), row[2]) for row in data]

    return render_template("report.html", data=formatted_data, raw_data=data, quick_tools=quick_tools_data, date=selected_date, user=session["user"])

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():

    selected_date = request.form['date']

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT model_name, prediction_value, created_at
    FROM predictions
    WHERE user_id = ? AND DATE(created_at) = ?
    """, (session["user"]["id"], selected_date))

    data = cursor.fetchall()

    # Fetch quick tools usage
    cursor.execute("""
    SELECT tool_name, created_at
    FROM quick_tools_usage
    WHERE user_id = ? AND DATE(created_at) = ?
    """, (session["user"]["id"], selected_date))

    quick_tools_data = cursor.fetchall()
    conn.close()

    # ===== Format prediction values =====
    def format_value(model_name, value):
        if "House" in model_name:
            # Format price as Cr or Lakh
            val = float(value)
            if val >= 100:
                return f"₹{val/100:.2f} Cr"
            else:
                return f"₹{val:.2f} Lakh"
        elif "Loan" in model_name:
            # Show as percentage
            return f"{float(value):.2f}%"
        elif "Fake" in model_name or "Spam" in model_name:
            # Show as percentage
            return f"{float(value):.2f}%"
        else:
            # Show as-is
            return f"{float(value):.2f}"

    pdf = FPDF()
    pdf.add_page()

    # Helper function to clean text for PDF
    def clean_text(text):
        """Remove problematic Unicode characters"""
        if not isinstance(text, str):
            text = str(text)
        # Remove non-ASCII characters that might cause issues
        return text.encode('ascii', errors='replace').decode('ascii')

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "AI SMART PREDICTION SYSTEM", ln=True, align="C")

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Prediction Analysis Report", ln=True, align="C")

    pdf.ln(5)

    # User Information
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "User Information", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    user = session.get("user")
    user_name = clean_text(user.get('name', 'N/A'))
    user_email = clean_text(user.get('email', 'N/A'))
    pdf.cell(0, 8, f"Name: {user_name}", ln=True)
    pdf.cell(0, 8, f"Email: {user_email}", ln=True)

    pdf.ln(3)

    # Date
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Report Date : {selected_date}", ln=True)

    pdf.ln(5)

    # Description
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        8,
        "This report is generated by the AI Smart Prediction System. "
        "The system uses trained Machine Learning models to analyse user inputs "
        "and generate predictions. These predictions help users understand patterns, "
        "probabilities and estimated outcomes."
    )

    pdf.ln(8)

    # Prediction Table Header
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, "Prediction Results", ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 10, "Model Name", 1)
    pdf.cell(40, 10, "Prediction", 1)
    pdf.cell(50, 10, "Time", 1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)

    # Table Data
    if data:
        for row in data:
            model = clean_text(str(row[0]))
            value = clean_text(format_value(row[0], row[1]))
            time = clean_text(str(row[2]))
            
            pdf.cell(60, 10, model[:28], 1)
            pdf.cell(40, 10, value if len(value) <= 15 else value[:12] + "...", 1)
            pdf.cell(50, 10, time[:18] if len(time) > 18 else time, 1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, "No predictions made on this date", ln=True)

    pdf.ln(10)

    # Quick Tools Usage Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, "Quick Tools Usage", ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(80, 10, "Tool Name", 1)
    pdf.cell(70, 10, "Usage Time", 1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)

    if quick_tools_data:
        for row in quick_tools_data:
            tool = clean_text(str(row[0]))
            time = clean_text(str(row[1]))
            
            pdf.cell(80, 10, tool[:35], 1)
            pdf.cell(70, 10, time[:30], 1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, "No quick tools used on this date", ln=True)

    pdf.ln(10)

    # AI Explanation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "AI Explanation", ln=True)

    pdf.set_font("Helvetica", "", 10)

    pdf.multi_cell(
        0,
        8,
        "The predictions above are generated using trained machine learning models. "
        "Each model analyses specific types of input data such as numerical values, "
        "text information or user attributes. The prediction values indicate the "
        "model's estimated output based on patterns learned during training. "
        "These results can help users make better decisions based on AI analysis. "
        "Quick tools usage shows the different AI utilities accessed on this date."
    )

    # Create PDF in memory (no file saving)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        download_name=f"AI_Report_{selected_date}.pdf",
        as_attachment=True
    )
# ================= USER DASHBOARD =================

@app.route("/user_dashboard")
def user_dashboard():
    # Allow a guest view of the dashboard so developers can open the page
    # without logging in (useful during development). If a real user is
    # in session, render with their data as before.
    user = session.get("user")
    if not user:
        user = {"id": None, "name": "Guest", "email": "", "photo": None}
    
    # Fetch prediction count for logged-in user
    prediction_count = 0
    if user and user.get("id"):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user["id"],))
        prediction_count = cursor.fetchone()[0]
        conn.close()
    
    return render_template("user_dashboard.html", user=user, prediction_count=prediction_count)

# ================= PROFILE PHOTO UPLOAD =================
@app.route("/upload_profile", methods=["POST"])
@login_required
def upload_profile():
    if "photo" not in request.files:
        return redirect("/user_dashboard")

    file = request.files["photo"]
    if file.filename == "":
        return redirect("/user_dashboard")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ✅ IMPORTANT: session update properly
    session["user"] = {
        "id": session["user"]["id"],
        "name": session["user"]["name"],
        "email": session["user"]["email"],
        "photo": filename
    }

    session.modified = True   # 🔥 THIS LINE FIXES IT

    return redirect("/user_dashboard")


# ================= PAGES (PROTECTED) =================
@app.route("/student")
@login_required
def student_page():
    return render_template("student.html")

@app.route("/house")
@login_required
def house_page():
    return render_template("house.html")

@app.route("/heart")
@login_required
def heart_page():
    return render_template("heart.html")

@app.route("/spam")
@login_required
def spam_page():
    return render_template("spam.html")

@app.route("/fake")
@login_required
def fake_page():
    return render_template("fake.html")

@app.route("/loan")
@login_required
def loan_page():
    return render_template("loan.html")

# ================= AI TOOLS  =================

@app.route("/sentiment")
@login_required
def sentiment():
    save_quick_tool_usage(session["user"]["id"], "Sentiment Analyzer")
    return render_template("sentiment_analyzer.html")

@app.route("/password")
@login_required
def password():
    save_quick_tool_usage(session["user"]["id"], "Password Checker")
    return render_template("password_checker.html")

@app.route("/word_counter")
@login_required
def word_counter():
    save_quick_tool_usage(session["user"]["id"], "Word Counter")
    return render_template("word_counter.html")

@app.route("/name_generator")
@login_required
def name_generator():
    save_quick_tool_usage(session["user"]["id"], "Name Generator")
    return render_template("name_generator.html")

@app.route("/summarizer")
@login_required
def summarizer():
    save_quick_tool_usage(session["user"]["id"], "Text Summarizer")
    return render_template("text_summarizer.html")

@app.route("/idea")
@login_required
def idea():
    save_quick_tool_usage(session["user"]["id"], "Idea Generator")
    return render_template("idea_generator.html")


# ================= CHATBOT =================

@app.route("/chatbot")
@login_required
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/chatbot_response", methods=["POST"])
@login_required
def chatbot_response():

    user_message = request.json.get("message", "").lower()

    # ================= GREETING =================
    if any(word in user_message for word in ["hello", "hi", "hey", "good morning", "good evening"]):
        reply = "Hello 👋 I am your AI Assistant. I can help you understand the system, prediction models, reports and more."

    # ================= HELP =================
    elif "help" in user_message:
        reply = (
            "I can help you with:\n"
            "• Available AI models\n"
            "• How predictions work\n"
            "• How to generate reports\n"
            "• Prediction history\n"
            "• System features\n"
            "Try asking: 'What models are available?'"
        )

    # ================= MODEL LIST =================
    elif "models" in user_message or "available models" in user_message:
        reply = (
            "This AI system contains multiple machine learning models:\n\n"
            "1️⃣ Student Score Prediction\n"
            "2️⃣ House Price Prediction\n"
            "3️⃣ Loan Approval Prediction\n"
            "4️⃣ Heart Disease Prediction\n"
            "5️⃣ Spam Email Detection\n"
            "6️⃣ Fake News Detection\n\n"
            "You can ask about any specific model."
        )

    # ================= STUDENT MODEL =================
    elif "student" in user_message or "exam" in user_message:
        reply = (
            "📚 Student Score Prediction Model\n\n"
            "This model predicts a student's exam score based on:\n"
            "• Study Hours\n"
            "• Attendance\n"
            "• Previous Marks\n\n"
            "It helps estimate student performance."
        )

    # ================= HOUSE PRICE MODEL =================
    elif "house" in user_message or "price" in user_message or "property" in user_message:
        reply = (
            "🏠 House Price Prediction Model\n\n"
            "This model predicts property price based on:\n"
            "• Number of BHK rooms\n"
            "• Area (square feet)\n"
            "• Location\n\n"
            "It helps estimate real estate market value."
        )

    # ================= LOAN MODEL =================
    elif "loan" in user_message:
        reply = (
            "💰 Loan Approval Prediction Model\n\n"
            "This model predicts whether a loan will be approved.\n\n"
            "Factors used:\n"
            "• Applicant income\n"
            "• Credit history\n"
            "• Loan amount\n"
            "• Employment status\n\n"
            "It helps banks evaluate loan eligibility."
        )

    # ================= HEART DISEASE MODEL =================
    elif "heart" in user_message or "heart disease" in user_message:
        reply = (
            "❤️ Heart Disease Prediction Model\n\n"
            "This model predicts the risk of heart disease based on medical data such as:\n"
            "• Age\n"
            "• Blood Pressure\n"
            "• Cholesterol Level\n"
            "• Maximum Heart Rate\n\n"
            "It helps identify potential heart disease risk."
        )

    # ================= SPAM EMAIL MODEL =================
    elif "spam" in user_message or "email" in user_message:
        reply = (
            "📧 Spam Email Detection Model\n\n"
            "This model classifies emails as:\n"
            "• Spam\n"
            "• Not Spam\n\n"
            "It uses Natural Language Processing (NLP) to analyze email text and detect unwanted messages."
        )

    # ================= FAKE NEWS MODEL =================
    elif "fake news" in user_message or "news" in user_message:
        reply = (
            "📰 Fake News Detection Model\n\n"
            "This model analyzes news articles and determines whether they are:\n"
            "• Fake News\n"
            "• Real News\n\n"
            "It uses Natural Language Processing and Machine Learning techniques."
        )

    # ================= REPORT SYSTEM =================
    elif "report" in user_message:
        reply = (
            "📊 AI Prediction Reports\n\n"
            "You can generate reports from the dashboard.\n\n"
            "Steps:\n"
            "1️⃣ Select a report date\n"
            "2️⃣ View predictions for that day\n"
            "3️⃣ Download PDF report\n"
            "4️⃣ View prediction graph"
        )

    # ================= HISTORY =================
    elif "history" in user_message or "past prediction" in user_message:
        reply = (
            "📁 Prediction History\n\n"
            "All predictions made by users are stored in the database.\n"
            "You can view past predictions and generate reports anytime."
        )

    # ================= QUICK TOOLS =================
    elif any(word in user_message for word in ["quick tools", "utilities", "tools", "helper", "useful tool"]):
        reply = (
            "⚡ Quick Tools & Utilities\n\n"
            "We offer useful AI-powered tools:\n\n"
            "📝 Text Summarizer\n"
            "Condense long texts instantly using NLP\n\n"
            "📊 Word Counter\n"
            "Analyze text statistics and word frequency\n\n"
            "🔐 Password Checker\n"
            "Check password strength and security\n\n"
            "💭 Sentiment Analyzer\n"
            "Analyze emotions and sentiments in text\n\n"
            "✨ Name Generator\n"
            "Generate creative names for projects, businesses\n\n"
            "💡 Idea Generator\n"
            "Brainstorm and generate innovative ideas\n\n"
            "Access these tools directly from your dashboard!"
        )

    # ================= FALLBACK =================
    else:
        reply = (
            "Sorry 😅 I didn't understand that.\n\n"
            "Try asking:\n"
            "• What models are available?\n"
            "• Explain loan prediction\n"
            "• How to generate report?\n"
            "• Tell me about quick tools"
        )

    return {"reply": reply}
# =================  STUDENT PREDICTIONS =================
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = int(request.form["attendance"])
    previous_marks = int(request.form["previous_marks"])

    # ===== Prediction =====
    score = student_model.predict([[
        study_hours,
        attendance,
        previous_marks
    ]])[0]

    save_prediction(session["user"]["id"], "Student Model", score)

    score = round(float(score), 2)

    # ===== Grade =====
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    else:
        grade = "D"

    # ===== Pass / Fail =====
    status = "PASS ✅" if score >= 40 else "FAIL ❌"

    # ===== Performance Label =====
    if score >= 80:
        performance = "Excellent 🌟"
    elif score >= 60:
        performance = "Good 👍"
    else:
        performance = "Average ⚠️"

    # ===== Improvement =====
    improvement = round(score - previous_marks, 2)

    # ===== Suggestion =====
    suggestions = []
    if study_hours < 2:
        suggestions.append("Increase daily study hours")
    if attendance < 75:
        suggestions.append("Improve attendance above 75%")
    if not suggestions:
        suggestions.append("Maintain consistency")

    return render_template(
        "student.html",
        prediction=score,
        grade=grade,
        status=status,
        performance=performance,
        improvement=improvement,
        suggestions=suggestions
    )


@app.route("/house_predict", methods=["POST"])
@login_required
def house_predict():
    # ===== Model prediction (IN LAKHS) =====
    price_lakhs = house_model.predict([[
        int(request.form["bhk"]),
        int(request.form["area"]),
        location_encoder.transform([request.form["location"]])[0]
    ]])[0]

    price_lakhs = round(float(price_lakhs), 2)

    area = int(request.form["area"])
    save_prediction(session["user"]["id"], "House Model", price_lakhs)

    # ===== Price Range (±10%) =====
    low_price_lakhs = round(price_lakhs * 0.9, 2)
    high_price_lakhs = round(price_lakhs * 1.1, 2)

    # ===== Price per sqft =====
    price_per_sqft = round((price_lakhs * 100000) / area, 2)

    # ===== Location Rating (simple logic) =====
    location = request.form["location"].lower()
    rating_map = {
        "bandra": "⭐⭐⭐⭐⭐",
        "andheri": "⭐⭐⭐⭐☆",
        "malad": "⭐⭐⭐⭐☆",
        "dadar": "⭐⭐⭐⭐☆",
        "borivali": "⭐⭐⭐☆☆",
        "thane": "⭐⭐⭐☆☆"
    }
    location_rating = rating_map.get(location, "⭐⭐⭐☆☆")

    # ===== Market Status =====
    if price_lakhs < 80:
        market_status = "🟢 Fair Deal"
    elif price_lakhs < 150:
        market_status = "🟡 Average"
    else:
        market_status = "🔴 Overpriced"

    # ===== 5-Year Future Value (8% growth) =====
    future_price_lakhs = round(price_lakhs * (1.08 ** 5), 2)

    # ===== Lakh / Crore Switch =====
    def format_price(val):
        if val >= 100:
            return round(val / 100, 2), "Cr"
        return val, "L"

    price, unit = format_price(price_lakhs)
    low_price, _ = format_price(low_price_lakhs)
    high_price, _ = format_price(high_price_lakhs)
    future_price, _ = format_price(future_price_lakhs)

    return render_template(
        "house.html",
        price=price,
        unit=unit,
        low_price=low_price,
        high_price=high_price,
        price_per_sqft=price_per_sqft,
        location_rating=location_rating,
        market_status=market_status,
        future_price=future_price
    )

@app.route("/spam_predict", methods=["POST"])
@login_required
def spam_predict():
    pred = spam_model.predict(
        spam_vectorizer.transform([request.form["email"]])
    )[0]
    result_value = 1 if pred else 0
    save_prediction(session["user"]["id"], "Spam Model", result_value)
    
    return render_template("spam.html", result="SPAM 🚨" if pred else "NOT SPAM ✅")

@app.route("/fake_predict", methods=["POST"])
@login_required
def fake_predict():
    # Get input text
    news_text = request.form["news"]
    
    # Transform text using TF-IDF
    X_vec = fake_vectorizer.transform([news_text])
    
    # Ensemble prediction: Average probabilities from both models
    if fake_nb_model is not None:
        # Use both LR and NB models (ensemble)
        lr_probs = fake_lr_model.predict_proba(X_vec)[0]
        nb_probs = fake_nb_model.predict_proba(X_vec)[0]
        
        # Average the probabilities
        probs = (lr_probs + nb_probs) / 2
    else:
        # Fallback to just LR model
        probs = fake_lr_model.predict_proba(X_vec)[0]
    
    # probs[0] = probability of REAL (class 0)
    # probs[1] = probability of FAKE (class 1)
    real_prob = probs[0] * 100
    fake_prob = probs[1] * 100
    
    # Determine result based on which probability is higher
    if fake_prob > real_prob:
        result = "FAKE 🔴"
        confidence = round(fake_prob, 2)
    else:
        result = "REAL 🟢"
        confidence = round(real_prob, 2)
    
    # Save prediction (store the confidence of the predicted class)
    save_prediction(session["user"]["id"], "Fake News Model", confidence)
    
    return render_template(
        "fake.html",
        result=result,
        confidence=confidence,
        real_probability=round(real_prob, 2),
        fake_probability=round(fake_prob, 2)
    )

@app.route("/loan_predict", methods=["POST"])
@login_required
def loan_predict():
    try:
        # Encode categorical variables
        gender = loan_encoders["person_gender"].transform([request.form["gender"].lower()])[0]
        education = loan_encoders["person_education"].transform([request.form["education"].lower()])[0]
        home_ownership = loan_encoders["person_home_ownership"].transform([request.form["home"].lower()])[0]
        loan_intent = loan_encoders["loan_intent"].transform([request.form["intent"].lower()])[0]
        previous_default = loan_encoders["previous_loan_defaults_on_file"].transform([request.form["previous_default"].lower()])[0]
        
        # Prepare features in correct order
        features_dict = {
            'person_age': int(request.form["age"]),
            'person_income': float(request.form["income"]),
            'person_emp_length': int(request.form["experience"]),
            'person_gender': gender,
            'person_education': education,
            'person_home_ownership': home_ownership,
            'loan_amnt': float(request.form["amount"]),
            'loan_intent': loan_intent,
            'loan_int_rate': float(request.form["interest"]),
            'loan_percent_income': float(request.form["percent_income"]),
            'cred_hist_length': int(request.form["credit_history"]),
            'person_credit_score': int(request.form["credit_score"]),
            'previous_loan_defaults_on_file': previous_default
        }
        
        # Create DataFrame for proper feature ordering
        import pandas as pd
        features_df = pd.DataFrame([features_dict])
        
        # Reorder columns to match training
        if hasattr(loan_lr_model, 'feature_names_in_'):
            features_df = features_df[loan_lr_model.feature_names_in_]
        
        # Scale features if scaler is available
        if loan_scaler is not None:
            numerical_cols = ['person_age', 'person_income', 'person_emp_length', 
                            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                            'cred_hist_length', 'person_credit_score']
            features_scaled = features_df.copy()
            features_scaled[numerical_cols] = loan_scaler.transform(features_df[numerical_cols])
        else:
            features_scaled = features_df
        
        # Ensemble prediction
        if loan_rf_model is not None and loan_gb_model is not None:
            # Use all three models (ensemble)
            lr_probs = loan_lr_model.predict_proba(features_scaled)[0]
            rf_probs = loan_rf_model.predict_proba(features_scaled)[0]
            gb_probs = loan_gb_model.predict_proba(features_scaled)[0]
            
            # Weighted average
            probs = (loan_weights[0] * lr_probs + 
                    loan_weights[1] * rf_probs + 
                    loan_weights[2] * gb_probs)
        else:
            # Fallback to single model
            probs = loan_lr_model.predict_proba(features_scaled)[0]
        
        # probs[0] = probability of NOT APPROVED (class 0)
        # probs[1] = probability of APPROVED (class 1)
        not_approved_prob = probs[0] * 100
        approved_prob = probs[1] * 100
        
        # Determine result based on which probability is higher
        if approved_prob > not_approved_prob:
            result = "APPROVED ✅"
            confidence = round(approved_prob, 2)
        else:
            result = "REJECTED ❌"
            confidence = round(not_approved_prob, 2)
        
        # Save prediction
        save_prediction(session["user"]["id"], "Loan Model", confidence)
        
        return render_template(
            "loan.html",
            result=result,
            probability=confidence,
            accuracy=loan_accuracy
        )
        
    except Exception as e:
        print(f"Error in loan_predict: {e}")
        return render_template(
            "loan.html",
            result="ERROR - Please check inputs",
            probability=0,
            accuracy=0
        )

@app.route("/heart_predict", methods=["POST"])
@login_required
def heart_predict():
    if heart_model is None or heart_encoders is None:
        return render_template("heart.html", result="Model unavailable", probability=0)

    try:
        # Encode categorical inputs
        gender = heart_encoders["Gender"].transform([request.form["gender"]])[0]
        smoking = heart_encoders["Smoking"].transform([request.form["smoking"]])[0]
        alcohol = heart_encoders["Alcohol Intake"].transform([request.form["alcohol"]])[0]
        family = heart_encoders["Family History"].transform([request.form["family_history"]])[0]
        diabetes = heart_encoders["Diabetes"].transform([request.form["diabetes"]])[0]
        obesity = heart_encoders["Obesity"].transform([request.form["obesity"]])[0]
        exercise_angina = heart_encoders["Exercise Induced Angina"].transform([request.form["exercise_angina"]])[0]
        chest_pain = heart_encoders["Chest Pain Type"].transform([request.form["chest_pain"]])[0]

        age = int(request.form.get("age", 0))
        cholesterol = float(request.form.get("cholesterol", 0))
        blood_pressure = float(request.form.get("blood_pressure", 0))
        heart_rate = float(request.form.get("heart_rate", 0))
        exercise_hours = float(request.form.get("exercise_hours", 0))
        stress = float(request.form.get("stress_level", 0))
        sugar = float(request.form.get("blood_sugar", 0))

        features = [[
            age, gender, cholesterol, blood_pressure, heart_rate,
            smoking, alcohol, exercise_hours, family, diabetes,
            obesity, stress, sugar, exercise_angina, chest_pain
        ]]

        prob = heart_model.predict_proba(features)[0][1] * 100
        prob = round(prob, 2)
        save_prediction(session["user"]["id"], "Heart Model", prob)

        # ---------------- RISK CATEGORY (4 LEVEL) ----------------
        if prob < 25:
            risk_category = "LOW RISK"
        elif prob < 50:
            risk_category = "MODERATE RISK"
        elif prob < 75:
            risk_category = "HIGH RISK"
        else:
            risk_category = "CRITICAL RISK"

        # ---------------- HEALTH SCORE ----------------
        health_score = 100 - prob
        health_score = round(health_score, 1)

        # ---------------- MAJOR RISK CONTRIBUTORS ----------------
        contributors = []

        if cholesterol > 240:
            contributors.append("High Cholesterol")
        if blood_pressure > 140:
            contributors.append("High Blood Pressure")
        if smoking == heart_encoders["Smoking"].transform(["Current"])[0]:
            contributors.append("Active Smoking")
        if diabetes == heart_encoders["Diabetes"].transform(["Yes"])[0]:
            contributors.append("Diabetes")
        if obesity == heart_encoders["Obesity"].transform(["Yes"])[0]:
            contributors.append("Obesity")
        if stress > 7:
            contributors.append("High Stress Level")
        if exercise_hours < 2:
            contributors.append("Low Physical Activity")

        if not contributors:
            contributors.append("No Major Clinical Risk Factors Detected")

        # ---------------- LIFESTYLE RECOMMENDATIONS ----------------
        recommendations = []

        if "High Cholesterol" in contributors:
            recommendations.append("Adopt low-fat & heart-friendly diet")
        if "High Blood Pressure" in contributors:
            recommendations.append("Reduce salt intake & monitor BP regularly")
        if "Active Smoking" in contributors:
            recommendations.append("Quit smoking immediately")
        if "Diabetes" in contributors:
            recommendations.append("Control blood sugar & regular monitoring")
        if "Obesity" in contributors:
            recommendations.append("Weight reduction through diet & exercise")
        if "High Stress Level" in contributors:
            recommendations.append("Practice meditation & stress management")
        if "Low Physical Activity" in contributors:
            recommendations.append("Exercise at least 30 mins daily")

        if not recommendations:
            recommendations.append("Maintain your current healthy lifestyle")

        return render_template(
            "heart.html",
            result=risk_category,
            probability=prob,
            health_score=health_score,
            contributors=contributors,
            recommendations=recommendations
        )

    except Exception as e:
        print("Error in heart_predict:", e)
        return render_template("heart.html", result="Error during prediction", probability=0)
    
    # ================= ANALYTICS =================
    
@app.route("/analytics")
@login_required
def analytics():

    conn = get_db()
    data = conn.execute("""
        SELECT model_name, prediction_value, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at ASC
    """, (session["user"]["id"],)).fetchall()
    conn.close()

    model_data = {}

    for row in data:
        model = row["model_name"]
        if model not in model_data:
            model_data[model] = {
                "dates": [],
                "values": []
            }

        model_data[model]["dates"].append(row["created_at"])
        model_data[model]["values"].append(row["prediction_value"])

    return render_template("analytics.html", model_data=model_data)
# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
