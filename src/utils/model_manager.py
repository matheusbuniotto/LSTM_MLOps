import torch
import os


class ModelManager:
    @staticmethod
    def save_model(model, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load_model(model_class, file_path, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(*args, **kwargs)
        model.load_state_dict(torch.load(file_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded from {file_path}")
        return model
