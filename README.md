# Modern AI Foundations

A collection of implementations exploring modern AI architectures and foundational models.

## Motivation

After reading key papers in robotics and AI:

- **SmolVLA**: Vision-Language-Action Model for Affordable Robotics
- **Diffusion Policy**: Visuomotor Policy Learning via Action Diffusion
- **ACT/ALOHA**: Learning Fine-grained Bimanual Manipulation
- **TDMPC**: Temporal Difference Learning for Model Predictive Control

I decided to implement different architectures to refresh my knowledge of SOTA approaches.

## Structure

- `vision_transformers/` - ViT variants (DiNAT, DeTR, MaskFormer, etc.)
- `visual_lang_models/` - Multimodal models (OWL-ViT, VQA, image captioning)
- `vae/` - Variational Autoencoders implementations, Conditional VAEs
- `intro_flow_match_diff_model/` - Flow matching and diffusion models

## References

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}

@article{chi2024diffusionpolicy,
  author = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  title ={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  journal = {The International Journal of Robotics Research},
  year = {2024},
}

@article{zhao2023learning,
  title={Learning fine-grained bimanual manipulation with low-cost hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}

@inproceedings{Hansen2022tdmpc,
  title={Temporal Difference Learning for Model Predictive Control},
  author={Nicklas Hansen and Xiaolong Wang and Hao Su},
  booktitle={ICML},
  year={2022}
}
```
