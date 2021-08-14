## Installation

sh setup.sh

## Run RetroFitting on W2V + WordNet
```Bash
conda activate retro
python wordnet_driver.py
```

This takes total 50 minutes for 2 iterations of retrofitting 56k Word2Vec vectors of dim 300 on WordNet lexicon.

![Timing](./images/progressbar.png)