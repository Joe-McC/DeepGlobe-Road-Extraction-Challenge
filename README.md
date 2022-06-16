# DeepGlobe-Road-Extraction-Challenge
Code for the 1st place solution in [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467).

This fork contains updated code and an conda environment.yml file to allow for use with Python 3 and updated pytorch versions. 


# Usage

### Data
Place '*train*', '*valid*' and '*test*' data folders in the '*dataset*' folder.

Data is from [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit). You should sign in first to get the data.

### Env 
- Setup Conda env using ```conda env create -f envrionment.yml```

### Train
- Run `python train.py` to train the default D-LinkNet34.

### Predict
- Run `python test.py` to predict on the default D-LinkNet34.

### Download trained D-LinkNet34
- [Dropbox](https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0)
- [百度网盘](https://pan.baidu.com/s/1wqyOEkw5o0bzbuj7gBMesQ)
