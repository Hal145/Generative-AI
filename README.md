# Generative-AI

## Class-Conditioned Diffusion Model Sampling
- Medical image sampling using DDIMSampler

| Class | Real Image | Synthetic Image |
|-------|------------|-----------------|
| lr    | ![lr_real](images/lr_image.jpeg) | ![lr_synthetic](images/lr_synthetic_image.jpeg) |
| ld    | ![ld_real](images/ld_image.jpeg) | ![ld_synthetic](images/ld_synthetic_image.jpeg) |
| mc    | ![mc_real](images/mc_image.jpeg) | ![mc_synthetic](images/mc_synthetic_image.jpeg) |
| ns    | ![ns_real](images/ns_image.jpeg) | ![ns_synthetic](images/ns_synthetic_image.jpeg) |


UMAP plot of synthetic and real image feature vectors. CONCH [Link Text](https://huggingface.co/MahmoodLab/CONCH) is used as a histopathology feature extractor.

![Alt text](images/umap_all_classes.jpeg)
