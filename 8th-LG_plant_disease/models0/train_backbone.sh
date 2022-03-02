python train_cnn.py --device cuda:0 \
                    --model 'tf_efficientnetv2_s' \
                    --lr 0.0005 \
                    --epochs 100 \
                    --base_folder 'C:\Users\bed1\src\dacon_farm\data\train' \
                    --label_fn 'C:\Users\bed1\src\dacon_farm\data\train.csv'