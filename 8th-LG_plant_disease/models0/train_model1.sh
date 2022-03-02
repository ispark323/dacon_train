TRAIN_DATA_FOLDER = 'C:\Users\bed1\src\dacon_farm\data\test' 
LABEL_FN = 'C:\Users\bed1\src\dacon_farm\data/sample_submission.csv'
RNN_FN = 'C:\Users\bed1\Desktop\dacon\checkpoint\0201101938_rnn\best_model.pth'

python train_model1.py --device cuda:0 \
                       --kfold_idx 0 \
                       --lr 1e-4 \ 
                       --epochs 100 \
                       --base_folder %{TRAIN_DATA_FOLDER} \
                       --label_fn %{LABEL_FN} \
                       --rnn_backbone %{RNN_FN} \

python train_model1.py --device cuda:0 \
                       --kfold_idx 1 \
                       --lr 1e-4 \ 
                       --epochs 100 \
                       --base_folder %{TRAIN_DATA_FOLDER} \
                       --label_fn %{LABEL_FN} \
                       --rnn_backbone %{RNN_FN} \

python train_model1.py --device cuda:0 \
                       --kfold_idx 2 \
                       --lr 1e-4 \ 
                       --epochs 100 \
                       --base_folder %{TRAIN_DATA_FOLDER} \
                       --label_fn %{LABEL_FN} \
                       --rnn_backbone %{RNN_FN} \

python train_model1.py --device cuda:0 \
                       --kfold_idx 3 \
                       --lr 1e-4 \ 
                       --epochs 100 \
                       --base_folder %{TRAIN_DATA_FOLDER} \
                       --label_fn %{LABEL_FN} \
                       --rnn_backbone %{RNN_FN} \

python train_model1.py --device cuda:0 \
                       --kfold_idx 4 \
                       --lr 1e-4 \ 
                       --epochs 100 \
                       --base_folder %{TRAIN_DATA_FOLDER} \
                       --label_fn %{LABEL_FN} \
                       --rnn_backbone %{RNN_FN} \