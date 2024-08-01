# Chinese Chess Detection Dataset

- Use [AnyLabeling](https://github.com/vietanhdev/anylabeling) to annotate all data in `data/data_01` and `data/data_02`.
- Run `make_data.sh` to generate the training data.

```bash
conda create -n chessai-dataprep python=3.9
conda activate chessai-dataprep
pip install -r requirements.txt
bash make_data.sh
```

## Citation

Please cite this paper if it helps your research:
```bibtex
@software{Nguyen_ChessAI_Data_-_2023,author = {Nguyen, Viet-Anh},doi = {10.5281/zenodo.1234},month = nov,title = {{ChessAI Data - Data for Chinese chess game position recognition}},url = {https://github.com/nrl-ai/chessai-data},version = {1.0.0},year = {2023}}
```
