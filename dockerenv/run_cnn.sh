python /developer/create_model_original.py
python /developer/create_model_categorical.py

python /developer/train_model.py model_categorical 5way 5way model_5way_test --epoch 3
python /developer/test_model_categorical.py model_5way_test 5way
