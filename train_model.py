
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Project folder path
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"


def train_model():

    study_hours = [
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10]
    ]

    student_scores = [
        35, 40, 50, 55, 60,
        68, 75, 80, 90, 95
    ]

    # Create ML model
    model = LinearRegression()

    # Train model
    model.fit(study_hours, student_scores)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("ML Model trained successfully")
    print(f"Model saved at: {MODEL_PATH}")


# Call function
if __name__ == "__main__":
    train_model()
