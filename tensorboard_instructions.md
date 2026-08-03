ssh -p 34211 -L 6006:127.0.0.1:6006 root@172.16.78.10

tensorboard --logdir logs/rsl_rl/g1_general_climbing --port 6006 --bind_all

OR for background:

nohup tensorboard --logdir logs/rsl_rl/g1_general_climbing --port 6006 --bind_all > /tmp/tb.log 2>&1 &

On laptop:  http://127.0.0.1:6006

