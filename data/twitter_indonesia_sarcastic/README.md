---
license: apache-2.0
language: 
- ind
pretty_name: "Twitter Indonesia Sarcastic"
---

# Twitter Indonesia Sarcastic

Twitter Indonesia Sarcastic is a dataset intended for sarcasm detection in the Indonesian language. This dataset is introduced in [Khotijah et al. (2020)](https://dl.acm.org/doi/10.1145/3406601.3406624), whereby Indonesian tweets are collected and labeled as either sarcastic or non-sarcastic. We took the [raw data](https://github.com/skhotijah/using-lstm-for-context-based-approach-of-sarcasm-detection-in-twitter/blob/main/dataset/Indonesia/imbalanced.csv), and performed several cleaning procedures such as: sentence order re-reversal, deduplication with minHash LSH, PII masking to remove usernames, hashtags, emails, URLs, and finally a random sampling to limit the non-sarcastic comments. Following [SemEval-2022 Task 6: iSarcasmEval](https://aclanthology.org/2022.semeval-1.111/), we used a 1:3 ratio to balance sarcastic with non-sarcastic comments.

## Dataset Structure

### Data Instances

```py
{
    'tweet': 'Terima kasih bapak <username> telah mengendalikan banjir dengan baik sehingga Jakarta saat ini tidak ada lagi yang tidak banjir.. Semua sudah merata.. ?????? <hashtag>',
    'label': 1
}
```

### Data Fields

- `tweet`: PII-masked Twitter tweet content.
- `label`: `0` for non-sarcastic, `1` for sarcastic.

### Data Splits

| Split                       | #sarcastic | #non sarcastic | #total |
| --------------------------- | :--------: | :------------: | :----: |
| `train`                     |    470     |      1408      |  1878  |
| `test`                      |    134     |      404       |  538   |
| `validation`                |     67     |      201       |  268   |
| Total (cleaned; balanced)   |    671     |      2013      |  2684  |
| Total (cleaned; unbalanced) |    671     |     12190      | 12861  |
| Total (raw)                 |    4350    |     13368      | 17718  |

### Dataset Directory

```sh
twitter_indonesia_sarcastic
├── README.md
├── data    # re-balanced dataset
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
└── raw_data
    ├── khotijah.csv    # raw dataset
    └── khotijah_cleaned.csv    # cleaned dataset
```

## Authors

Twitter Indonesia Sarcastic is prepared by:

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