# 训练
`llamafactory-cli train examples/train_full/qwen3_1___7B_full_sft.yaml`

爆内存的话，增加一个内存整理：

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
llamafactory-cli train examples/train_full/train_qwen3_14B_full_sft.yaml`

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
llamafactory-cli train examples/train_full/train_qwen3_14B_full_sft_afsim.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
llamafactory-cli train examples/train_lora/qwen3_32B_lora_sft_afsim.yaml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
llamafactory-cli train examples/train_full/train_qwen3_8B_full_sft_afsim.yaml



# 推理
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/qwen3_1.7B_full_sft.yaml

LOCAL_RANK=0 llamafactory-cli chat examples/inference/qwen3_14B_sft.yaml


# 模型转换
llamafactory-cli export /root/code/Llama-Factory/examples/merge_lora/qwen3_lora_sft_afsim.yaml

# llamafactory 训练完以后显存占用异常
pkill -15 -f 'llamafactory'

# VLLM后端
`pip install vllm -i https://mirrors.aliyun.com/pypi/simple`
如果提示没有gcc，请安装gcc
`apt install -y build-essential`

# Vllm启动
CUDA_VISIBLE_DEVICES=1,2,3 vllm serve /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B/afsim_sft_0714 --max-model-len 32768 --tensor-parallel-size 4 --gpu-memory-utilization 0.9  --host 0.0.0.0 --port 11400 --api-key sk-123456 --served-model-name afsim-qwen-32b

vllm serve /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B/afsim_sft_0714 --max-model-len 32768 --tensor-parallel-size 3 --gpu-memory-utilization 0.9  --host 0.0.0.0 --port 11400 --api-key sk-123456 --served-model-name afsim32b

vllm serve /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B/afsim_sft_0719 --max-model-len 32768 --tensor-parallel-size 2 --gpu-memory-utilization 0.8  --host 0.0.0.0 --port 11400 --api-key sk-123456 --served-model-name afsim-32b-0719

vllm serve /mnt/saves-wu-nas/shifan/kaifeng/models/Qwen3-32B --max-model-len 32768 --tensor-parallel-size 4  --gpu-memory-utilization 0.75  --host 0.0.0.0 --port 11400 --api-key sk-123456 --served-model-name qwen3-32b

vllm serve /root/code/Llama-Factory/saves/Qwen3-8B/full/sft_0728_afsim --max-model-len 32768 --tensor-parallel-size 4  --gpu-memory-utilization 0.75  --host 0.0.0.0 --port 11400 --api-key sk-123456 --served-model-name afsim8b

## for 53开发机
CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/ht_g3/saves-wu-nas/shifan/kaifeng/models/Qwen3-8B --max-model-len 32768 --tensor-parallel-size 1 --gpu-memory-utilization 0.9  --host 0.0.0.0 --port 8256 --api-key sk-123456 --served-model-name qwen-8b

---
# evalscope 

安装：
conda create -n evalscope python=3.10
conda activate evalscope
pip install uv
uv pip install 'evalscope[app]' -i https://mirrors.aliyun.com/pypi/simple

evalscope eval \
    --model /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B/afsim_sft_0714 \
    --datasets gsm8k

CUDA_VISIBLE_DEVICES=0 evalscope eval \
    --model /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B \
    --datasets gsm8k
    --limit 200

CUDA_VISIBLE_DEVICES=1 evalscope eval \
    --model /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B \
    --datasets mmlu
    --limit 200

CUDA_VISIBLE_DEVICES=2 evalscope eval \
    --model /mnt/saves-wu-nas/shifan/kaifeng/models/DistillQwen-ThoughtY-32B/afsim_sft_0714 \
    --datasets mmlu
    --limit 200

下面不行
evalscope eval \
    --model models/afsim32b \
    --api-url http://jb-jupyter-9079258978245500928-8080-nhss-job.z2120.nhss.zhejianglab.com:32119/v1 \
    --api-key sk-123456
    --eval-type service \
    --datasets gsm8k mmlu \
    --limit 5

