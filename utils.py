import torch
from pathlib import Path



#funkcja umożliwiająca zapiasnie modelu
def save_model(model:torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    # Save the model
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model, f=model_save_path)

#funkcja umożliwiająca wgranie modelu
def load_model(model_name: str):
    model_path = "./models/" + model_name
    loaded_model = torch.load(f=model_path)
    return loaded_model
