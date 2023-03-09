# Hugo

## Tools Used

### Fine-grained Image Captioning with CLIP Reward

* Authors: [Jaemin Cho](https://j-min.io), [David Seunghyun Yoon](https://david-yoon.github.io/), [Ajinkya Kale](https://www.linkedin.com/in/kaleajinkya/), [Franck Dernoncourt](https://research.adobe.com/person/franck-dernoncourt), [Trung Bui](https://sites.google.com/site/trungbuistanford/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
* [Findings of NAACL 2022 Paper](https://arxiv.org/abs/2205.13115)
* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/j-min/CLIP-Caption-Reward/blob/main/Inference_example.ipynb) (Inference using pretrained model on custom image)
* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/NAACL2022/CLIP-Caption-Reward)
* Try Replicate web demo and docker image here [![Replicate](https://replicate.com/j-min/clip-caption-reward/badge)](https://replicate.com/j-min/clip-caption-reward)
<img src="./assets/teaser.png" alt="teaser image" width="800"/>



## Acknowledgments
We thank the developers of [CLIP-ViL](https://github.com/clip-vil/CLIP-ViL/tree/master/CLIP-ViL-Direct/caption), [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch), [CLIP](https://github.com/openai/CLIP), [coco-caption](https://github.com/tylin/coco-caption), [cider](https://github.com/vrama91/cider) for their public code release.


## Reference
Please cite our paper if you use our models in your works:


```bibtex
@inproceedings{Cho2022CLIPReward,
  title     = {Fine-grained Image Captioning with CLIP Reward},
  author    = {Jaemin Cho and Seunghyun Yoon and Ajinkya Kale and Franck Dernoncourt and Trung Bui and Mohit Bansal},
  booktitle = {Findings of NAACL},
  year      = {2022}
}
