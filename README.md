# Trading Calculator with ML Predictions

A modern web application that helps traders calculate risk/reward ratios and get machine learning-powered predictions for their trades.

## Features

- Calculate risk/reward ratios
- Position sizing based on account size
- Risk management calculations
- Machine learning predictions for trade outcomes
- Modern, responsive UI with dark mode
- Real-time calculations

## Tech Stack

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python Flask
- Machine Learning: scikit-learn
- Data Processing: pandas, numpy
- Deployment: Render

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd trading-calculator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python model_trainer.py
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5001`

## Deployment to Render

1. Create a GitHub repository and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: trading-calculator
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free (or choose a paid plan for better performance)

## Project Structure

- `app.py` - Flask backend server
- `model_trainer.py` - ML model training script
- `index.html` - Main web interface
- `style.css` - Styling
- `script.js` - Frontend logic
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Notes

- The free tier of Render will spin down after 15 minutes of inactivity
- The ML model is trained on dummy data for demonstration purposes
- For production use, replace the dummy data with real trading data

## License

MIT License