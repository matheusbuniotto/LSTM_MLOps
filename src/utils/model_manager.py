import json
import os

import torch


class ModelManager:
    @staticmethod
    def save_model(model, file_path, model_args):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Salva pesos do modelo
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

        # Salva metadata
        metadata_path = file_path.replace(".pth", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(model_args, f, indent=4)
        print(f"Model metadata saved to {metadata_path}")

    @staticmethod
    def load_model(model_class, file_path):
        import json

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metadata_path = file_path.replace(".pth", "_metadata.json")
        with open(metadata_path, "r") as f:
            model_args = json.load(f)

        model = model_class(**model_args)
        model.load_state_dict(torch.load(file_path, map_location=device))
        model = model.to(device)
        model.eval()

        print(f"Model loaded from {file_path}")
        return model
