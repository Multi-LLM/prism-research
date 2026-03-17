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
git clone -b prism/shm https://github.com/ovg-project/kvcached.git
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
pip install -e . --no-build-isolation
python setup.py build_ext --inplace
```
