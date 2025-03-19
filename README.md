# TULIP: Towards Unified Language-Image Pre-training

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)  [![arXiv](https://img.shields.io/badge/arXiv-2311.16090-red)](https://arxiv.org/abs/2312.08366)

Check out our [project page](https://tulip-berkeley.github.io) for more information/the latest news and links!

This repository is a fork of [OpenCLIP](https://github.com/mlfoundations/open_clip), integrating the **TULIP** model. OpenCLIP is an open-source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training).

**This code only supports inference of the TULIP models. Training code will be released soon.**

## Features

- **OpenCLIP Foundation**: Supports various CLIP models trained on datasets like LAION-400M, LAION-2B, and DataComp-1B.
- **TULIP Enhancements**:
  - Optimized vision-language/vision-vision/language-language contrastive learning.
  - Generative Augmentation
- **Pretrained Models**: Includes models from OpenCLIP and TULIP training pipelines.
- **Zero-shot and Few-shot Learning**: Supports evaluation across multiple datasets.

## Installation

To install OpenCLIP + TULIP, use the following commands:

```sh
pip install timm --upgrade
pip install transformers
pip install -e .
```

## Model Checkpoints

The following models are currently available for inference:

Model Name | Resolution | Checkpoint
--- | --- | ---
TULIP-B-16-224 | 224 | Coming Soon
TULIP-H-14-224 | 224 | Coming Soon
TULIP-so400m-14-384 | 384 | [Download](https://s3.us-west-1.wasabisys.com/tulip/tulip-so400m-14-384.ckpt)
TULIP-G-16-384 | 384 | [Download](https://s3.us-west-1.wasabisys.com/tulip/tulip-G-16-384.ckpt)

## Inference

You can use TULIP for inference with the following code snippet:

```python
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('TULIP-so400m-14-384', pretrained='<path to model checkpoint>')
model.eval()

image = preprocess(Image.open("sample.jpg")).unsqueeze(0)
tokenizer = open_clip.get_tokenizer('TULIP-so400m-14-384')
text = tokenizer(["a cat", "a dog", "a bird"])

with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probabilities:", similarities)
```

## License

This project follows a modified licensing structure. Please refer to the [LICENSE](LICENSE) file for full details.

## Citation

If you use TULIP in your research, please cite the following paper:

```
@misc{tang2025tulip,
  title     = {TULIP: Towards Unified Language-Image Pretraining},
  author    = {Zineng Tang and Long Lian and Seun Eisape and XuDong Wang and Roei Herzig and Adam Yala and Alane Suhr and Trevor Darrell and David M. Chan},
  institution = {University of California, Berkeley},
  year      = {2025},
  note      = {Preprint},
}
```

## Acknowledgments

This project builds upon OpenCLIP and contributions from various research groups. Special thanks to:
- The OpenCLIP team for providing a robust framework.
- The TULIP development team for enhancements.

For issues or contributions, please open a GitHub issue or submit a pull request.
