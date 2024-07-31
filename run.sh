#generate dataset
python ./preprocess/dataset_generate.py

#Detect, generate keywords
python ./preprocess/keyword_detect.py

#Investigate
python ./preprocess/google_search.py
python ./preprocess/news_encode.py

#Judge
python ./model/dem_inv.py
python ./model/google_inv.py

#Determine
python ./model/determine.py