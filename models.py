import os
import joblib

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.health_model = None
        self.failure_model = None
        self.vehicle_encoder = None
        self.health_encoder = None
        self.failure_encoder = None

    def load_models(self):
        """Loads all the necessary models and encoders from the models directory."""
        health_model_path = os.path.join(self.models_dir, "health_model.pkl")
        failure_model_path = os.path.join(self.models_dir, "failure_model.pkl")
        vehicle_encoder_path = os.path.join(self.models_dir, "vehicle_encoder.pkl")
        health_encoder_path = os.path.join(self.models_dir, "health_encoder.pkl")
        failure_encoder_path = os.path.join(self.models_dir, "failure_encoder.pkl")

        if not all(os.path.exists(p) for p in [
            health_model_path, failure_model_path, vehicle_encoder_path,
            health_encoder_path, failure_encoder_path
        ]):
            raise FileNotFoundError("One or more model files are missing in the models directory.")

        self.health_model = joblib.load(health_model_path)
        self.failure_model = joblib.load(failure_model_path)
        self.vehicle_encoder = joblib.load(vehicle_encoder_path)
        self.health_encoder = joblib.load(health_encoder_path)
        self.failure_encoder = joblib.load(failure_encoder_path)

model_manager = ModelManager()
