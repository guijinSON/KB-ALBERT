# ZIP
This repository contains code for ZIP, a Zero-Shot Classifier for Financial Risk Tolerance Probing. This project was conducted as a part of [Future Finance A.I. Challenge](https://2021-icfss.org/?page_id=203) organized by KB Financial Group Inc.

## Project Explanation
This project cites the framework proposed by [Shashank Srivastava](https://aclanthology.org/P18-1029/) to transform a Pretrained Language Model(PLM) fine-tuned on a Natural Language Inference(NLI) dataset into a Zero-Shot Classifier. The output model, ZIP, is used to infer Financial Risk Tolerance by comprehending the investment opinions of eash individuals. ZIP leverages a pretrained Language Model [KB-ALBERT](https://github.com/KB-AI-Research/KB-ALBERT) fine-tuned on [KorNLI](https://github.com/kakaobrain/KorNLUDatasets), a Korean Natural Language Inference(NLI) dataset, as text encoder for the Zero-Shot Topic Classification Pipeline.   

<p align="middle">
  <img width="446" height="233" src="https://raw.githubusercontent.com/guijinSON/ZIP/main/assets/title.png">
</p> 

Awarded <strong> 6th place </strong> at Future AI Challenge(KB Bank), 2021.

## Demo
Make sure you have the adequate dependencies installed before running the demo.   
Path for model weights should be modified beforehand. 
```python

from ZIP.inference import ZIP
ZIP('적금, 주식, 펀드 등에 분산 투자를 통해 손실 위험을 최대한 회피한다.',tokenizer,model,labels=['분산 투자','집중 투자'])

```

<p align="left">
  <img width="600" height="180" src="https://raw.githubusercontent.com/guijinSON/ZIP/main/assets/demo_1.png">
</p>

## Contributors
-  강주연
-  손규진 | [GUIJIN SON](https://github.com/guijinSON)
-  최예린 

## Acknowledgements 
Project ZIP was conducted as a part of Future AI Challenge (KB Bank), 2021.   
Special thanks to [KB-AI-Research](https://github.com/KB-AI-Research/KB-ALBERT) for providing the pretrained KB-ALBERT model and [KakaoBrain](https://github.com/kakaobrain/KorNLUDatasets) for making Korean NLI datasets publicly available.



## References

```bibtex
@inproceedings{srivastava-etal-2018-zero,
    title = "Zero-shot Learning of Classifiers from Natural Language Quantification",
    author = "Srivastava, Shashank  and
      Labutov, Igor  and
      Mitchell, Tom",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1029",
    doi = "10.18653/v1/P18-1029",
    pages = "306--316",
    abstract = "Humans can efficiently learn new concepts using language. We present a framework through which a set of explanations of a concept can be used to learn a classifier without access to any labeled examples. We use semantic parsing to map explanations to probabilistic assertions grounded in latent class labels and observed attributes of unlabeled data, and leverage the differential semantics of linguistic quantifiers (e.g., {`}usually{'} vs {`}always{'}) to drive model training. Experiments on three domains show that the learned classifiers outperform previous approaches for learning with limited data, and are comparable with fully supervised classifiers trained from a small number of labeled examples.",
}
```

```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
