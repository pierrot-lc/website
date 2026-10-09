Models can be trained on large sizes, using their best decoding method, ours are evaluated on mixed precision.

| Model               | TSP-100  |        | TSP-250  |        | TSP-500  |        | TSP-1000  |        | TSP-10000 |        | TSPLIB 1~100 |         |TSPLIB 101~1000 |         |TSPLIB 1001~10000 |        |
|:--------------------|:--------:|:------:|:-------: |:------:|:--------:|:------:|:---------:|:------:|:---------:|:------:|:------------:|:-------:|---------------:|:-------:|-----------------:|:------:|
|                     | _Gap_    | _Time_ | _Gap_    | _Time_ | _Gap_    | _Time_ | _Gap_     | _Time_ | _Gap_     | _Time_ | _Gap_        |  _Time_ | _Gap_          |  _Time_ | _Gap_            | _Time_ |
| BQ-NCO              | 0.08     | 0.5m   | 0.17     | 2.3m   | 0.58     | 6.8m   | 1.43      | 16m    | 20.94     | 51m    |              |         |                |         |                  |        |
| LEHD                | 0.01     | 11.8m  | 0.01     | 29m    | 0.35     | 60m    | 1.17      | 123m   |           |        |              |         |                |         |                  |        |
| ELG                 |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| PENS-AR             |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| L2C                 |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| INViT-3V            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| DGL                 |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| Fast T2T            | 1.04     | 1.0m   | 0.60     | 1.4m   | 0.69     | 3.4m   | 0.80      | 8.8m   | 1.48      | 4.9m   |              |         |                |         |                  |        |
| SIL                 | 1.71     | 2.0m   | 1.18     | 5.1m   | 1.01     | 10.6m  | 1.26      | 22m    | 2.01      | 21m    |              |         |                |         |                  |        |
| COExpander          | 0.02     | 0.3m   | 0.15     | 1.2m   | 0.29     | 2.4m   | 0.62      | 5m     | 1.29      | 14m    |              |         |                |         |                  |        |
| UDC                 | 0.35     | 3.3m   | 2.16     | 2.0m   | 1.56     | 3.6m   | 1.77      | 21m    | OOM       | OOM    |              |         |                |         |                  |        |
| MaskCO              | 0.01     | 0.2m   | 0.01     | 0.3m   | 0.01     | 0.3m   | 0.01      | 0.5m   | 674       | 7m     |              |         |                |         |                  |        |
| NIR-1024 RRC(100)   | 0.20     | 0.4m   | 0.06     | 0.4m   | 0.08     | 1.3m   | 0.18      | 6m     | 2.14      | 3.5m   | 0.43         | 0.0m    | 0.37           | 0.5m    | 1.79             | 4.0m   |
| NIR-1024 RRC(1000)  | 0.03     | 0.9m   | 0.01     | 3.4m   | 0.03     | 14.2m  | 0.11      | 69m    | 1.58      | 31m    | 0.29         | 0.1m    | 0.28           | 4.3m    |                  |        |
| NIR-1024 RRC(10000) |          |        |          |        |          |        |           |        | 1.29      | 291m   |              |         |                |         | 1.11             | 375m   |

| Model      | Decoding method                                        |
|:-----------|:------------------------------------------------------:|
| BQ-NCO     | Beam search (16) & k-NN (250)                          |
| LEHD       | Random reconstruct (100)                               |
| SIL        | Random reconstruct (100)                               |
| COExpander | S=4, Ds=3, Is=5, 2-OPT                                 |
| NIR        | Random reconstruct (10000) & 2-OPT 100 & Parallel (10) |

NOTE: RRC is done with alpha = 1000. When the instances have lower cities than 1000, I turn to pure
parallel best-of-N sampling.

[Fast T2T]: https://arxiv.org/abs/2502.02941
[UDC]: https://arxiv.org/abs/2407.00312
[GenSCO]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b8e9869fa827226ff68db9290659a20c-Abstract-Conference.html
