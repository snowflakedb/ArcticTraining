mkdir -p results
deepspeed -H /data-fast/hostfile -i 10.4.141.181@10.4.141.110 /code/users/jrasley/all_reduce_bench.py &> results/0.log &
deepspeed -H /data-fast/hostfile -i 10.4.131.30@10.4.142.46 /code/users/jrasley/all_reduce_bench.py &> results/1.log &
deepspeed -H /data-fast/hostfile -i 10.4.139.227@10.4.133.86 /code/users/jrasley/all_reduce_bench.py &> results/2.log &
deepspeed -H /data-fast/hostfile -i 10.4.140.242@10.4.128.146 /code/users/jrasley/all_reduce_bench.py &> results/3.log &
