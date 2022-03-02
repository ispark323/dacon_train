#!/bin/bash
START_TIME=$(date +%s)

TEST_DATA_FOLDER='./data/test'
LABEL_FN='./data/sample_submission.csv'

python model_001.py --base_folder $TEST_DATA_FOLDER \
                    --label_fn $LABEL_FN \
                    --weight_folder './weights/0202113411' \

python model_002.py --base_folder $TEST_DATA_FOLDER \
                    --label_fn $LABEL_FN \
                    --weight_folder './weights/0203180850'

python model_003.py --base_folder $TEST_DATA_FOLDER \
                    --weight_fn './weights/submission_0201_001/best_model.pth'

python run_ensemble.py

END_TIME=$(date +%s)
echo "It took $(($END_TIME - $START_TIME)) seconds to inference..."