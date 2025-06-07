`main.py` assumes query answer scores are stored in `answers`. Here an example of the structure expected for QTO: 


```
answers/qto
└── FB15k237+H
    ├── 1p
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    ├── 2i
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    ├── 2p
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    ├── 3i
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    ├── 3p
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    ├── ip
    │   ├── qto_scores.pkl
    │   └── query_answer_ranks.pkl
    └── pi
        ├── qto_scores.pkl
        └── query_answer_ranks.pkl
```
