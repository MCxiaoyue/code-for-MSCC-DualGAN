import os
cmd = 'python main.py --phase test --dataset_name 256voicedataset --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100'
os.system(cmd)
cmd = 'python main2.py --phase test --dataset_name 256voicedataset --image_size 256 --lambda_B 1000.0 --lambda_C 1000.0 --epoch 100'
os.system(cmd)
