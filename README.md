<h1 align="center">Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation</h1> 

<p align="center">
<a href="https://arxiv.org/abs/2401.17904"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

This is the official repository of the paper [Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation](https://arxiv.org/abs/2401.17904).

## :sparkles: Highlight

![overview](.asset/overview.png)

- **Hierarchical Text Segmentation.** Hi-SAM unifies text segmentation across stroke, word, text-line, and paragraph levels. Hi-SAM also achieves layout analysis as a by-product.
- **High-Quality Text Stroke Segmentation & Stroke Labeling Assistant.** High-quality text stroke segmentation by introducing mask feature of 1024×1024 resolution with minimal modification in SAM's original mask decoder. 
- **Automatic and Interactive.** Hi-SAM supports both automatic mask generation and interactive promptable mode. Given a single-point prompt, Hi-SAM provides word, text-line, and paragraph masks.

## :bulb: Overview of Hi-SAM
![Hi-SAM](.asset/Hi-SAM.png)

## :label: TODO 

- [ ] Release demo.
- [ ] Release training and inference codes.

## 💗 Acknowledgement

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [HierText](https://github.com/google-research-datasets/hiertext), [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [TextSeg](https://github.com/SHI-Labs/Rethinking-Text-Segmentation)

## :black_nib: Citation

```bibtex
@article{ye2024hi-sam,
  title={Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation},
  author={Ye, Maoyuan and Zhang, Jing and Liu, Juhua and Liu, Chenyu and Yin, Baocai and Liu, Cong and Du, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2401.17904},
  year={2024}
}
```