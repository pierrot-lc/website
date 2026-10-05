Models can be trained on large sizes, using their best decoding method, ours are evaluated on mixed precision.

| Model          | TSP-100  |        | TSP-250  |        | TSP-500  |        | TSP-1000  |        | TSP-10000 |        | TSPLIB 1~100 |         |TSPLIB 101~1000 |         |TSPLIB 1001~10000 |        |
|:---------------|:--------:|:------:|:-------: |:------:|:--------:|:------:|:---------:|:------:|:---------:|:------:|:------------:|:-------:|---------------:|:-------:|-----------------:|:------:|
|                | _Gap_    | _Time_ | _Gap_    | _Time_ | _Gap_    | _Time_ | _Gap_     | _Time_ | _Gap_     | _Time_ | _Gap_        |  _Time_ | _Gap_          |  _Time_ | _Gap_            | _Time_ |
| BQ-NCO         | 0.08     | 0.5m   | 0.17     | 2.3m   | 0.58     | 6.8m   | 1.43      | 16m    | 20.94     | 51m    |              |         |                |         |                  |        |
| LEHD           | 0.01     | 11.8m  | 0.01     | 29m    | 0.35     | 60m    | 1.17      | 123m   |           |        |              |         |                |         |                  |        |
| ELG            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| PENS-AR        |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| L2C            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| INViT-3V       |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| DGL            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| Fast T2T       |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| MaskCO         |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| SIL            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| COExpander     |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| UDC            |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| GenSCO         |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |
| MaskCO         |          |        |          |        |          |        |           |        |           |        |              |         |                |         |                  |        |

| Model  | Decoding method               |
|:-------|:-----------------------------:|
| BQ-NCO | Beam search (16) & k-NN (250) |
| LEHD   | Random reconstruct (100)      |

[Fast T2T]: https://arxiv.org/abs/2502.02941
[UDC]: https://arxiv.org/abs/2407.00312
[GenSCO]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b8e9869fa827226ff68db9290659a20c-Abstract-Conference.html
