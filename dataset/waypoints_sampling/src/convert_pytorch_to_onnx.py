import torch
import torch.onnx
import yaml

from waypoints_sampling.model.util import load_model

if __name__ == "__main__":

    model_path = 'checkpoints/vanilla-vae-94-20230729.ckpt'
    model_type = 'vae'
    model_config_file = 'config/vae.yaml'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    with open(model_config_file, 'r') as file:
        config = yaml.safe_load(file)

    model = load_model(model_path=model_path,
                    model_type=model_type,
                    model_config=config)
    model = model.to(device)
    print(type(model))
    print(device)

    full_model_output_path = 'vanilla_vae_model.onnx'
    
    obs_img_tensor = torch.randn(1, 18, 64, 85, dtype=torch.float32, device=device)
    goal_img_tensor = torch.randn(1, 3, 64, 85, dtype=torch.float32, device=device)
    

    torch.onnx.export(model=model,
                    args=(obs_img_tensor, goal_img_tensor),               
                    f = f"{full_model_output_path}",
                    opset_version=10
                    )
    print(f"The ONNX model has been created successfully..")











    





