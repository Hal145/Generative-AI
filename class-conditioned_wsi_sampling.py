import warnings
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
warnings.filterwarnings("ignore")
import csv

device = torch.device('cuda:0')

r""" Class-conditioned diffusion model sampling  """

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def get_model(config_path, device, checkpoint):
    config = OmegaConf.load(config_path)
    del config['model']['params']['first_stage_config']['params']['ckpt_path']
    del config['model']['params']['unet_config']['params']['ckpt_path']
    model = load_model_from_config(config, checkpoint, device)
    model_device = model.device
    return model, model_device

def get_unconditional_token(batch_size, device):
    return {"class_label": torch.tensor(batch_size * [0]).to(device)}

def get_conditional_token(batch_size, device, class_label):
    batch = {
    "class_label": torch.tensor(batch_size * [class_label]).to(device)}
    return batch

if __name__ == "__main__":
    # Set up model checkpoints and configuration files
    ckpt_path = "path/to/model_checkpoint"
    config_path = "path/to/model_configuration_file"

    model, model_device = get_model(config_path, device, ckpt_path)
    sampler = DDIMSampler(model)

    # Parameters for generation
    batch_size = 16
    shape = [3, 64, 64]
    scale = 1.5  # Scale of classifier-free guidance - parameter of the diffusion model
    samples_per_bag = 5000 # number of samples per bag
    bags_per_class = 100 # number of whole slide images that will be created for each class
    classes = 4
    class_names = ["ns", "mc", "lr", "ld"]

    # Directory to save images
    output_dir = Path("/media/nfs/generated_dataset")
    output_dir.mkdir(exist_ok=True)

    csv_data = []

    for class_label in range(1,4): #classes
        print(f"Generating patches for class {class_label}")
        class_name = class_names[class_label]
        for bag_idx in range(bags_per_class):
            bag_dir = output_dir / f"synthetic_chl_{class_name}_{bag_idx}"
            bag_dir.mkdir(parents=True, exist_ok=True)
            instances_generated = 0

            while instances_generated < samples_per_bag:
                with torch.no_grad():
                    ut = get_unconditional_token(batch_size, model_device)
                    uc = model.get_learned_conditioning(ut)

                    ct = get_conditional_token(batch_size, model_device, class_label)
                    cc = model.get_learned_conditioning(ct)

                    samples_ddim, _ = sampler.sample(50, batch_size, shape, cc, verbose=False, \
                                                     unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=0)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()

                    for i, sample in enumerate(x_samples_ddim):
                        if instances_generated >= samples_per_bag:
                            break
                        # Convert generated images to PIL image and save as JPEG
                        img = Image.fromarray(sample.permute(1, 2, 0).numpy(), 'RGB')
                        img.save(bag_dir / f"image_{instances_generated}.jpg")
                        instances_generated += 1

            # Log directory and class label in CSV file
            csv_data.append([str(bag_dir), class_label])

    # Write generated whole slide image names to a CSV file
    csv_file = output_dir / "generated_data.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Directory", "Label"])
        writer.writerows(csv_data)

    print("All patches generated and saved successfully.")
