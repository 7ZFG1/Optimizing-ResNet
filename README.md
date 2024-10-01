Train and Test Resnet Models
-----------------------------
Train and validate resnet models. You can also use optimized resnet models in ```model.py```.
Mlflow is integrated into ```train.py``` to observe/compare the performance of trained models.
If you have yolo dataset, ```convert_dataset.py``` can be used to convert it to the relevant format. 


Installation
-----
```
pip3 install -r requirements.txt
```

Train
-----
```
python3 train.py --model_name OptimizedResnet101 --batch_size 32 --num_epochs 100 --train_data_path "" --val_data_path "" --model_save_path "" --device cuda:0
```

Test
-----
```
python3 test.py --model_name OptimizedResnet101 --model_path "" --batch_size 32 --test_data_path "" -result_save_path "" --device cuda:0
```
