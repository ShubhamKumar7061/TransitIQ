# ⬡ TransitIQ — Multi-Modal Transport Demand Prediction

A stunning, ML-powered transport demand prediction web app built with Python + Flask.

### Step 1 — Open in VS Code
```
File → Open Folder → select `transport-demand-prediction`
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```
Works in **Chrome**, **Microsoft Edge**, Firefox, etc.

---

## 🧠 ML Models Used

| Model | Purpose |
|-------|---------|
| `GradientBoostingRegressor` | Predicts passenger demand count |
| `RandomForestClassifier` | Classifies preferred transport mode |

## 📦 Project Structure

```
transport-demand-prediction/
├── app.py                  ← Flask backend + ML models
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Main web page
└── static/
    ├── css/style.css       ← Dark futuristic design
    └── js/main.js          ← Charts, animations, API calls
```

## ✨ Features

- 12-parameter journey input form
- Real-time Gradient Boosting demand prediction
- Random Forest transport mode classifier
- Hourly demand trend chart (Chart.js)
- Cross-mode comparison bar chart
- AI-generated recommendations
- Fully responsive (mobile + desktop)
- Animated KPI cards with live counters
- Smooth scroll reveal animations
- Dark futuristic UI with glassmorphism
