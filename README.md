# ZIP: Zero-Shot Financial Risk Tolerance Classifier 
ZIP is a Zero-Shot Financial Risk Tolerance Classifier designed to infer the investment behavior of individuals.   
It leverages a pretrained Language Model [KB-ALBERT](https://github.com/KB-AI-Research/KB-ALBERT) fine-tuned on [KorNLI](https://github.com/kakaobrain/KorNLUDatasets), a Korean Natural Language Inference(NLI) dataset, as text encoder for the Zero-Shot Topic Classification Pipeline. 

## 사용 방법      

1. pip install requirements.txt -> 필요한 패키지 설치
2. demo.ipynb 실행 -> 파워 포인트에 담겨 있는 실행 사진들은 모두 해당 주피터 노트북에 기록되어 있습니다. 

## Contributors
-  강주연
-  손규진 | [GUIJIN SON](https://github.com/guijinSON)
-  최예린 

## Acknowledgements 
Project ZIP was conducted as a part of Future AI Challenge (KB Bank), 2021.   
Special thanks to [KB-AI-Research](https://github.com/KB-AI-Research/KB-ALBERT) for providing the pretrained KB-ALBERT model.

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
