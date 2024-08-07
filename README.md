# IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection

This project aims to benchmark and evaluate various language models for sarcasm detection in Indonesian. We experiment with classical machine learning models, fine-tuned transformer models, and zero-shot classification using large language models. All of our models, datasets, and results are openly available via HuggingFace Hub.

<div align="center">

<a href="https://huggingface.co/collections/w11wo/indonesian-sarcasm-detection-65840069489f3b53a0452c04"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collections-yellow"></img></a>

</div>

## Pre-trained Models

| Base Model             | #params | Reddit                                                                                                        | Twitter                                                                                                         |
| ---------------------- | :-----: | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| IndoNLU IndoBERT Base  |  124M   | [IndoNLU IndoBERT Base Reddit](https://huggingface.co/w11wo/indobert-base-p1-reddit-indonesia-sarcastic)      | [IndoNLU IndoBERT Base Twitter](https://huggingface.co/w11wo/indobert-base-p1-twitter-indonesia-sarcastic)      |
| IndoNLU IndoBERT Large |  335M   | [IndoNLU IndoBERT Large Reddit](https://huggingface.co/w11wo/indobert-large-p1-reddit-indonesia-sarcastic)    | [IndoNLU IndoBERT Large Twitter](https://huggingface.co/w11wo/indobert-large-p1-twitter-indonesia-sarcastic)    |
| IndoLEM IndoBERT Base  |  111M   | [IndoLEM IndoBERT Base Reddit](https://huggingface.co/w11wo/indobert-base-uncased-reddit-indonesia-sarcastic) | [IndoLEM IndoBERT Base Twitter](https://huggingface.co/w11wo/indobert-base-uncased-twitter-indonesia-sarcastic) |
| mBERT Base             |  178M   | [mBERT Base Reddit](https://huggingface.co/w11wo/bert-base-multilingual-cased-reddit-indonesia-sarcastic)     | [mBERT Base Twitter](https://huggingface.co/w11wo/bert-base-multilingual-cased-twitter-indonesia-sarcastic)     |
| XLM-R Base             |  278M   | [XLM-R Base Reddit](https://huggingface.co/w11wo/xlm-roberta-base-reddit-indonesia-sarcastic)                 | [XLM-R Base Twitter](https://huggingface.co/w11wo/xlm-roberta-base-twitter-indonesia-sarcastic)                 |
| XLM-R Large            |  560M   | [XLM-R Large Reddit](https://huggingface.co/w11wo/xlm-roberta-large-reddit-indonesia-sarcastic)               | [XLM-R Large Twitter](https://huggingface.co/w11wo/xlm-roberta-large-twitter-indonesia-sarcastic)               |

## Dataset

We used two datasets for training and evaluation, including a novel dataset of Reddit comments and a Twitter dataset. The Reddit dataset consists of 14,116 comments, while the Twitter dataset consists of 12,861 tweets.

| Dataset                     | Link                                                                             |
| --------------------------- | -------------------------------------------------------------------------------- |
| Reddit Indonesia Sarcastic  | [HuggingFace](https://huggingface.co/datasets/w11wo/reddit_indonesia_sarcastic)  |
| Twitter Indonesia Sarcastic | [HuggingFace](https://huggingface.co/datasets/w11wo/twitter_indonesia_sarcastic) |

## Results

We compared the performance of various models on both the Reddit and Twitter datasets. The evaluation metric used is the F1-score.

| Model                    | Reddit F1-score | Twitter F1-score |
| ------------------------ | :-------------: | :--------------: |
| **Classical**            |                 |                  |
| Logistic Regression      |     0.4887      |      0.7142      |
| Naive Bayes              |     0.4591      |      0.6721      |
| SVC                      |     0.4467      |      0.6782      |
| **Fine-tuning**          |                 |                  |
| IndoBERT Base (IndoNLU)  |     0.6100      |      0.7273      |
| IndoBERT Large (IndoNLU) |     0.6184      |      0.7160      |
| IndoBERT Base (IndoLEM)  |     0.5671      |      0.6462      |
| mBERT                    |     0.5338      |      0.6467      |
| XLM-R Base               |     0.5690      |      0.7386      |
| XLM-R Large              |   **0.6274**    |    **0.7692**    |
| **Zero-shot**            |                 |                  |
| BLOOMZ-560M              |     0.3870      |      0.3916      |
| BLOOMZ-1.1B              |     0.3944      |      0.3987      |
| BLOOMZ-1.7B              |     0.3758      |      0.3885      |
| BLOOMZ-3B                |     0.4000      |      0.3847      |
| BLOOMZ-7.1B              |     0.4036      |      0.3968      |
| mT0 Small                |     0.4000      |      0.3988      |
| mT0 Base                 |     0.3990      |      0.3985      |
| mT0 Large                |     0.3998      |      0.3989      |
| mT0 XL                   |     0.4001      |      0.3988      |

## Citation

If you use this work in your research, please cite:

```bibtex
@article{10565877,
  author = {Suhartono, Derwin and Wongso, Wilson and Tri Handoyo, Alif},
  journal = {IEEE Access}, 
  title = {IdSarcasm: Benchmarking and Evaluating Language Models for Indonesian Sarcasm Detection}, 
  year = {2024},
  volume = {12},
  number = {},
  pages = {87323-87332},
  keywords = {Social networking (online);Blogs;Machine learning;Feature extraction;Accuracy;Deep learning;Electronic mail;Natural language processing;Sentiment analysis;Low-resource data;low-resource languages;Indonesian sarcasm detection;natural language processing;sarcasm detection;sentiment analysis},
  doi = {10.1109/ACCESS.2024.3416955}
}
```

## Author

<a href="https://github.com/w11wo">
    <img src="https://github.com/w11wo.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 1px #fff;margin:0 4px;">
</a>

## References

```bibtex
@inproceedings{10.1145/3406601.3406624,
    author = {Khotijah, Siti and Tirtawangsa, Jimmy and Suryani, Arie A.},
    title = {Using LSTM for Context Based Approach of Sarcasm Detection in Twitter},
    year = {2020},
    isbn = {9781450377591},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3406601.3406624},
    doi = {10.1145/3406601.3406624},
    booktitle = {Proceedings of the 11th International Conference on Advances in Information Technology},
    articleno = {19},
    numpages = {7},
    keywords = {context, Sarcasm detection, paragraph2vec, lstm, deep learning},
    location = {, Bangkok, Thailand, },
    series = {IAIT '20}
}

@article{Ranti2020IndonesianSD,
    title={Indonesian Sarcasm Detection Using Convolutional Neural Network},
    author={Kiefer Stefano Ranti and Abba Suganda Girsang},
    journal={International Journal of Emerging Trends in Engineering Research},
    year={2020},
    url={https://doi.org/10.30534/ijeter/2020/10892020}
}

@article{academicReddit,
    title= {Reddit comments/submissions 2005-06 to 2023-09},
    journal= {},
    author= {stuck_in_the_matrix, Watchful1, RaiderBDev},
    year= {},
    url= {},
    abstract= {Reddit comments and submissions from 2005-06 to 2023-09 collected by pushshift and u/RaiderBDev. These are zstandard compressed ndjson files. Example python scripts for parsing the data can be found here https://github.com/Watchful1/PushshiftDumps},
    keywords= {reddit},
    terms= {},
    license= {},
    superseded= {}
}

@inproceedings{abu-farha-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 6: i{S}arcasm{E}val, Intended Sarcasm Detection in {E}nglish and {A}rabic",
    author = "Abu Farha, Ibrahim  and
      Oprea, Silviu Vlad  and
      Wilson, Steven  and
      Magdy, Walid",
    editor = "Emerson, Guy  and
      Schluter, Natalie  and
      Stanovsky, Gabriel  and
      Kumar, Ritesh  and
      Palmer, Alexis  and
      Schneider, Nathan  and
      Singh, Siddharth  and
      Ratan, Shyam",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.111",
    doi = "10.18653/v1/2022.semeval-1.111",
    pages = "802--814",
}
```