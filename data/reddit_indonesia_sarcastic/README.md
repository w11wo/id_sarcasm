---
license: apache-2.0
language: 
- ind
pretty_name: "Reddit Indonesia Sarcastic"
---

# Reddit Indonesia Sarcastic

Reddit Indonesia Sarcastic is a dataset intended for sarcasm detection in the Indonesian language. This dataset is inspired by the data collection procedure introduced in [Ranti, K.S., & Girsang, A.S (2020)](http://www.warse.org/IJETER/static/pdf/file/ijeter10892020.pdf), whereby Reddit comments from r/indonesia subreddit are collected and filtered by the existence of an `/s` tag at the end of the comment. We collected Reddit comments from 2020-01 to 2023-09 from [Academic Torrents](https://academictorrents.com/details/89d24ff9d5fbc1efcdaf9d7689d72b7548f699fc) and applied the aforementioned procedure. Further, we performed deduplication with minHash LSH, PII masking to remove usernames, hashtags, emails, URLs, and finally a random sampling to limit the non-sarcastic comments. Following [SemEval-2022 Task 6: iSarcasmEval](https://aclanthology.org/2022.semeval-1.111/), we used a 1:3 ratio to balance sarcastic with non-sarcastic comments.

## Dataset Structure

### Data Instances

```py
{
    "author": "curuya",
    "created_utc": 1584876528,
    "score": 7,
    "permalink": "/r/indonesia/comments/fmxhfe/jangan_takut_sama_corona_takut_sama_allah/fl6n993/",
    "subreddit": "indonesia",
    "body": 'taat perintah tuhan : "kalau ada razia mendingan kabur" /s',
    "lang_fastText": "id",
    "label": 1,
    "text": 'taat perintah tuhan : "kalau ada razia mendingan kabur"',
}
```

### Data Fields

- `author`: Comment author.
- `created_utc`: Comment creation time, in UTC. 
- `score`: Comment's Reddit voting score.
- `permalink`: Permalink to the Reddit comment. 
- `subreddit`: Subreddit name.
- `body`: Raw Reddit comment content.
- `lang_fastText`: Language detected by [fasttext-langdetect](https://github.com/zafercavdar/fasttext-langdetect).
- `label`: `0` for non-sarcastic, `1` for sarcastic.
- `text`: Sarcastic tag-removed, PII-masked version of `body`.

### Data Splits

| Split              | #sarcastic | #non sarcastic | #total  |
| ------------------ | :--------: | :------------: | :-----: |
| `train`            |    2470    |      7411      |  9881   |
| `test`             |    706     |      2118      |  2824   |
| `validation`       |    353     |      1058      |  1411   |
| Total (balanced)   |    3529    |     10587      |  14116  |
| Total (unbalanced) |    3529    |    2616335     | 2619864 |

### Dataset Directory

```sh
reddit_indonesia_sarcastic/
├── README.md
├── data    # re-balanced dataset
│   ├── test.json
│   ├── train.json
│   └── validation.json
└── raw_data    # raw unbalanced dataset
    └── reddit_indonesia_sarcastic.json
```

## Authors

Reddit Indonesia Sarcastic is prepared by:

<a href="https://github.com/w11wo">
    <img src="https://github.com/w11wo.png" alt="GitHub Profile" style="border-radius: 50%;width: 64px;border: solid 1px #fff;margin:0 4px;">
</a>

## References

```bibtex
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