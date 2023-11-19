# Data Preparation for ChessAI

- Use [AnyLabeling](https://github.com/vietanhdev/anylabeling) to annotate all data in `data/data_01` and `data/data_02`.
- Run `make_data.sh` to generate the training data.

```bash
conda create -n chessai-dataprep python=3.9
conda activate chessai-dataprep
pip install -r requirements.txt
bash make_data.sh
```
