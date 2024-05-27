from clize import run
from autofaiss import build_index

def main(
    *,
    embeddings="/output/emb", 
    index_path="/output/index/knn.index",
    index_infos_path="/output/index/index_infos.json",
    max_index_memory_usage="50G",
    current_memory_available="60G",
    nb_cores=16,
):
    build_index(
        embeddings=embeddings, 
        index_path=index_path,
        index_infos_path=index_infos_path, 
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        metric_type="ip",
        nb_cores=nb_cores,
    )

if __name__ == "__main__":
    run(main)
