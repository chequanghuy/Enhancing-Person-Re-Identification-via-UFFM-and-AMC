# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on,t,c

python3 tools/test.py --config_file='configs/market1501.yml' --seed 0 --k 3 --n_triple 600 

