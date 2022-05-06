# CAISE
Code and dataset for AAAI 2022 paper ["CAISE: Conversational Agent for Image Search and Editing"](https://arxiv.org/abs/2202.11847) Hyounghun Kim, Doo Soon Kim, Seunghyun Yoon, Franck Dernoncourt, Trung Bui, and Mohit Bansal

## Prerequisites

- Python 3.8
- PyTorch 1.10
```
pip install -r requirements.txt
```
Please download [this](https://drive.google.com/file/d/14nfsyaVeQITu7qneeAbxVb7xCwk4zZ3u/view?usp=sharing) and put it in dataset/multidial folder.

## Usage
```
bash run_main.sh INDEX_GPU
```
INDEX_GPU is the index of the gpu you want to run the model on.


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
