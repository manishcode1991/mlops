# DVC init 

# parameter of dvc 

# create pipeline into dvc 


dvc run --force \
-n generate_test_data \
-d src/generate_text_data.py  \
-p hyper -p train_set_no_of_records -p train_set_limit \
python src/generate_text_data.py

dvc run --force \
-n fit_create_model \
-d src/fit_create_model.py  \
python src/fit_create_model.py