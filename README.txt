My code can be accessed at: git@github.com:jacarr4/supervised-learning.git
There are several required python packages: numpy, pandas, scikit-learn. There may be others.

There are several scripts of interest. They are arranged by problem order in the documentation.
* python -m number_1 --dataset <dataset> --clustering <clustering>
* python -m number_2 --dataset <dataset>
* python -m number_3 --dataset <dataset> --clustering <clustering>
* python -m number_4 --dataset <dataset>
* python -m number_5 --dataset <dataset>

dataset can be one of: Digits, PimaIndians
clustering can be one of: kmeans, em

Additionally, python -m kmeans_digits will generate the visualization plots I used in my analysis. 
To change the dataset for these, edit line 20: Dataset.Digits -> Dataset.PimaIndians

That is all. Thank you!

Jacob Carr
