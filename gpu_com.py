from llm_inference_simulator import GPUCatalog, get_gpu, ClusterSpec

# 방법 1: GPUCatalog 사용
gpu = GPUCatalog.get_gpu("A10-24GB")

# 방법 2: 편의 함수 사용 (더 짧음!)
gpu = get_gpu("A10")

# 방법 3: 유연한 이름
gpu = get_gpu("h100")  # H100-80GB를 찾아줌
gpu = get_gpu("b200")  # B200-192GB

# ClusterSpec에 바로 사용
cluster = ClusterSpec(
    n_gpus_per_node=1,
    gpu_spec=get_gpu("A10")
)

# GPU 목록 확인
print(GPUCatalog.list_available())

# GPU 비교
GPUCatalog.compare(["A10-24GB", "A100-80GB", "H100-80GB", "B200-192GB"])

# 모든 GPU 보기
GPUCatalog.print_all()

# 세대별 GPU
hopper_gpus = GPUCatalog.get_by_generation("hopper")
