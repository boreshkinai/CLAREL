# zeroshot-pairwise

NV_GPU=0,1 nvidia-docker run -p 1234:8888 -p 1235:6006 -p 1236:6007 -p 1237:6008 -p 1238:6009 -v /mnt/scratch/boris:/mnt/scratch/boris -v /mnt/datasets/public/:/mnt/datasets/public/ -v /mnt/home/boris:/mnt/home/boris -t -d --name boris_zeroshot_exp boris_zeroshot
