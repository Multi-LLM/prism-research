## Install

### Start redis server

Start redis server with docker:
```bash
docker run --name my-redis -p 6379:6379 -d redis
```

Install redis-py with `pip install redis`.

Test the redis server is running:
```bash
>>> import redis
>>> r = redis.Redis(host='localhost', port=6379, db=0)
>>> r.set('foo', 'bar')
True
>>> r.get('foo')
b'bar'
```

### Download sglang-mulit-model and kvcached
```bash
# install sglang-muli-model
git clone https://github.com/Multi-LLM/prism-research.git

# install kvcached
git clone https://github.com/ovg-project/kvcached.git
```

### Start sglang development docker container

1. Start a container running in the background.
```bash
docker run -dit --gpus all --ipc=host --network=host \
    -v `pwd`/prism-research:/sgl-workspace/sglang-multi-model/ \
    -v `pwd`/kvcached:/sgl-workspace/kvcached \
    -v ~/.cache/huggingface/:/root/.cache/huggingface \
    --name dev-sglang-{your_name} \
    lmsysorg/sglang:v0.3.4.post2-cu121 bash
```
If you are using H100 from the SGLang team, change the mount directory of `.cache`.
```bash
docker run -dit --gpus all --ipc=host --network=host \
    -v `pwd`/prism-research:/sgl-workspace/sglang-multi-model/ \
    -v `pwd`/kvcached:/sgl-workspace/kvcached \
    -v /opt/dlami/nvme/.cache:/root/.cache \
    --env "HF_TOKEN={your_huggingface_token}" \
    --name dev-{your_name} \
    lmsysorg/sglang:v0.3.4.post2-cu121 bash
```

2. Login to the container.
```bash
docker exec -it dev-sglang bash
```

### Install sglang-multi-model and kvcached

```bash
cd /sgl-workspace/sglang-multi-model
pip install -e "python[all]"
# flashinfer has already been installed in the container
# pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.4/

cd /sgl-workspace/kvcached
pip install -e .
```

## Run multi-model tests

### Single model 

Launch server with model_path:
```bash
cd benchmark/multi-model
python3 -m sglang.launch_multi_model_server --port 30000 --disable-cuda-graph --model-config-file ./model_configs/model_config_single.json --disable-radix-cache --enable-elastic-memory --use-kvcached-v0 --log-file ./server.log
```

 Options:
    
    - `enable-elastic-memory`: enable elastic memory with kvcached. Should not be enabled for now. 
    - `policy`: The policy for multi-model scheduling.
    - `enable-controller`: Whether to use multi-model scheduler to switch on/off models.
    - `log-file`: The server log file path. If specified, the server log will be saved to the file.

Run tests:
```bash
cd benchmark/multi-model
python3 benchmark.py -n 1
```

### Multiple models

Multiple models need to be launched with a model config file.

See [swap_2.json](./model_configs/swap_2.json) and [collocate.json](./model_configs/collocate.json) for examples.

Note that: 
- The `model_name` should be unique.
- `max_memory_pool_size` is the maximum memory pool size in GB. It should be set according to the GPU memory size. A utils function `get_memory_pool_size`(../../python/sglang/multi_model/utils/get_memory_pool_size.py) is provided to get the memory pool size.
- `on` is a boolean value. If `on` is true, the instance's model will be placed on GPU and be ready to serve generation requests. If `on` is False, the instance's model will be initially placed on CPU, and waiting for a later `Activate` request, which will then load the model to GPU and start serving generation requests.
- For the swapping case, the model with `on` set to false is suggested to be put before the model with `on` set to true in the config file, to avoid running out of GPU memory when initializing the memory pool for the `on` = false models.

Launch server with

```bash
cd benchmark/multi-model
python3 -m sglang.launch_multi_model_server --model-config-file ./model_configs/swap_2.json  --disable-cuda-graph --disable-radix-cache --max-mem-usage 67.28 --enable-controller --enable-cpu-share-memory --enable-elastic-memory --use-kvcached-v0 --policy priority-multi-gpu --log-file ./server.log --async-loading
```

Options: 
 - `--enable-controller`: Wheterh to use multi-model scheduler to switch on/off models.
 - `max-mem-usage`: Optional. The GPU memory that can be used for model weights and memory pool size, e.g. total GPU memory * 0.85.

Then waiting for the server to be ready, which may take a few minutes, depending on the number of models. All models that are configured to be on will be fired up and ready to roll!å

### Run tests(Arena trace)

First uncomment or write the experiments you want to run in `def test_all(self)` and then run the following command:

For non-tp setting (an example using 8 gpus and 18 models): 

```bash
python3 -m sglang.launch_multi_model_server \
  --log-file server-logs/e2e.log\
  --model-config-file model_configs/8_gpu_18_model_our.json \
  --enable-cpu-share-memory \
  --disable-cuda-graph \
  --disable-radix-cache \
  --use-kvcached-v0 \
  --max-mem-usage 67.28 \
  --port 30333 \
  --enable-elastic-memory \
  --enable-gpu-scheduler \
  --enable-controller \
  --policy simple-global \
  --enable-model-service \
  --enable-worker-pool \
  --workers-per-gpu 4 \
  --num-model-service-workers 4 \
  --num-gpus 8
```

```bash
python3 benchmark.py \
  --base-url http://127.0.0.1:30333 \
  --real-trace ./real_trace.pkl \
  --num-models 18 \
  --num-gpus 8 \
  --exp-name our_test \
  --e2e-benchmark \
  --time-scale 1 \
  --replication 1
```

### Run tests(TP)

For tp setting (an example using 4 gpus and 2 models):

Starting the server:
```bash
python3 -m sglang.launch_multi_model_server \
  --model-config-file ./model_configs/4_gpu_2_model.json \
  --host 127.0.0.1 --port 33333 \
  --disable-cuda-graph --disable-radix-cache \
  --log-file server-logs/$(date +%m%d_%H%M)_ours_4_gpu_tp.log \
  --load-format dummy \
  --enable-elastic-memory --use-kvcached-v0 \
  --policy simple-global
```

Run test:
```bash
python3 benchmark.py \
  --base-url http://127.0.0.1:33333 \
  --exp-name ours_4_gpu \
  --num-gpus 4 \
  --num-models 2 \
  --model-paths model_1 model_4 \
  --time-scale 0.5 \
  --replication 1 \
  --e2e-benchmark \
  --real-trace ./real_trace.pkl \
  --results-path benchmark-results \
  --request-path output-requests \
  --seed 42 \
  --memory-pool-size 16 \
  --req-rate 20
``` 
