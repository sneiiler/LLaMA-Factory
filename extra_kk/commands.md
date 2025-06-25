# 训练
`llamafactory-cli train examples/train_full/qwen3_1___7B_full_sft.yaml`

# 推理
`CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/qwen3_1.7B_full_sft.yaml` 

# VLLM后端
`pip install vllm -i https://mirrors.aliyun.com/pypi/simple`
如果提示没有gcc，请安装gcc
`apt install -y build-essential`

