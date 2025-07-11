import json
import random

with open("/mnt/saves-wu-nas/shifan/kaifeng/DistilQwen_100k/distil_qwen_100k.json") as f:
    result = json.load(f)

subset_data = random.sample(result, 50000)

with open("/mnt/saves-wu-nas/shifan/kaifeng/DistilQwen_100k/distil_qwen_50k_subset.json", "w") as f:
    json.dump(subset_data, f,ensure_ascii=False,indent=4)