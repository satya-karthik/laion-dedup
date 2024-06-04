from PIL import Image
from glob import glob
from clize import run
import os
from pathlib import Path
import torch
import faiss
import numpy as np
import joblib
from feature_extractor import create_model
import webdataset as wds
from compute_embeddings import log_and_continue
import tarfile
from tqdm import tqdm


def extract_file_from_tar(tar_path: str, file_name: str,
                          dest_folder: str, save_file_name: str = None) -> None:
    """
    Extracts a specific file from a tar archive and saves it to the designated folder.

    Args:
        tar_path (str): The path to the tar archive.
        file_name (str): The name of the file to extract from the tar archive.
        dest_folder (str): The folder where the extracted file will be saved.
        save_file_name (str): The name of the file to be saved in the dest_folder.
        defaults to None. if no value was provided file_name will be used.


    Returns:
        None
    """
    tar_path = Path(tar_path)
    dest_folder = Path(dest_folder)

    # Ensure destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        # Extract the specified file
        member = tar.getmember(file_name)
        tar.extract(member, dest_folder)

        # Move the extracted file to the root of the destination folder
        if save_file_name is not None:
            extracted_path = dest_folder / member.name
            final_path = dest_folder / Path(save_file_name)
            extracted_path.rename(final_path)


class MetaData:

    def __init__(self, folder):
        self.folder = folder

    def build(self):
        self.paths = list(sorted(glob(os.path.join(self.folder, "*.npy"))))
        self.sizes = []
        for path in self.paths:
            data = np.load(path, allow_pickle=True)
            self.sizes.append(len(data))

    def get_indices(self, indices):
        return [self.get(ind) for ind in indices]

    def get(self, index):
        # start, end, path_index = self.path_index[index]
        # path = self.paths[path_index]
        # data = np.load(path, allow_pickle=True)
        # offset = index - start
        # return data[offset]

        # avg = self.cumsum[-1] / len(self.cumsum)
        # pos = int(index // avg)
        # while pos < len(self.cumsum) and pos >= 0:
        # start = 0 if pos == 0 else self.cumsum[pos-1]
        # end = self.cumsum[pos]
        # print(index, pos, start, end)
        # # print(pos, start, end, index)
        # if start <= index < end:
        # path = self.paths[pos]
        # data = np.load(path, allow_pickle=True)
        # offset = index - start
        # return data[offset]
        # elif index < start:
        # pos -= 1
        # else:
        # pos += 1
        # raise ValueError()
        # nb = 0
        start = 0
        for path, size in zip(self.paths, self.sizes):
            end = start + size
            if start <= index < end:
                data = np.load(path, allow_pickle=True)
                offset = index - start
                return data[offset]
            start = end


def expand2square(pil_img, background_color):
    # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def resize(image, tf):
    image = tf(image)
    return image


@torch.no_grad()
def main(
    *,
    dataset_name="Laion_tar_2",
    root="root",
    cosine_similarity_threshold=0.604169,
    split="test",
    index="/output/index/knn.index",
    metadata="/output/meta",
    batch_size=128,
    num_workers=4,
    model_config="dedup_seer1.5B.th",
    out="/output/results/exp_9_t_0_604169",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_slug = dataset_name.replace("/", "_")

    # Load metadata and index
    print("Loading metadata and index...")
    if os.path.isdir(metadata):
        metadata = MetaData(metadata)
        metadata.build()
        joblib.dump(metadata, os.path.join(os.path.dirname(index), "meta.pkl"))
    else:
        metadata = joblib.load(metadata)

    # metadata.cumsum = np.cumsum(metadata.sizes)
    start = 0
    metadata.path_index = []
    for i, size in enumerate(metadata.sizes):
        end = start + size
        metadata.path_index.extend([(start, end, i)] * size)
        start = end

    image_index = faiss.read_index(index)
    weight_name = 'checkpoint_0009'
    model, preprocessor = create_model(
        weight_name=weight_name,
        device='cuda', model_dir="/weights")
    model = model.to(device)
    model.eval()

    data_list = map(
        str, Path("/data/").resolve().glob("*.tar"))
    pipeline = [wds.SimpleShardList(data_list)]
    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
    ])
    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png"),
        wds.map_dict(image=preprocessor),
        wds.to_tuple("image", "json"),
        wds.batched(32, partial=True),
    ])
    dataset = wds.DataPipeline(*pipeline)
    ds = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
    )
    features_cache = os.path.join(
        out, f"{dataset_name.replace('/','_')}_{index.replace('/','_')}_{weight_name.replace('/', '_')}.npy")
    meta_cache = os.path.join(
        out, f"{dataset_name.replace('/','_')}_{index.replace('/','_')}_{weight_name.replace('/', '_')}_meta.npy")
    print(features_cache)
    print(meta_cache)
    # create the folder if doesn't exists.
    Path(features_cache).resolve().parent.mkdir(parents=True, exist_ok=True)

    image_features_list = []
    meta_features = []
    count = 0
    print("Computing image features...")
    for X, meta in tqdm(ds, unit="batches"):
        file_meta = [{"tar_file": f"/data/{x['key'][:-4]}.tar",
                      "file_name": f"{x['key']}.jpg"} for x in meta]
        X = X.to(device)

        image_features = model(X)
        image_features = image_features.data.cpu()
        image_features = image_features.view(len(image_features), -1)
        image_features_list.append(image_features)
        meta_features.append(file_meta)

        image_features = np.concatenate(image_features_list)

        print("Performing range search...")

        D, I_ = image_index.search(image_features, 1)
        print("Score:", D.mean())
        L, D, I_ = image_index.range_search(
            image_features, cosine_similarity_threshold)

        assert len(L) - 1 == len(image_features)
        print("Start..")
        for i in range(len(L) - 1):
            indices = I_[L[i]: L[i + 1]]
            dists = D[L[i]: L[i + 1]]
            if len(indices):

                order = np.argsort(-dists)
                indices = indices[order][0:10]
                dists = dists[order][0:10]
                if dists.size > 1:
                    count += 1
                    print(f"count: {count}", end="\r")

                    file_meta = meta_features[0][i]
                    folder = os.path.join(
                        out, str(Path(file_meta["file_name"]).stem))
                    # extract the original image
                    extract_file_from_tar(tar_path=file_meta["tar_file"],
                                          file_name=file_meta["file_name"],
                                          dest_folder=folder,
                                          save_file_name="query_image.jpg")

                    # get all the similar images.
                    meta_files = metadata.get_indices(indices)
                    for i, file_meta in enumerate(meta_files):
                        save_file_name = f"{file_meta['file_name'][:-4]}_{str(dists[i]).replace('.','_')}_{file_meta['file_name'][-4:]}"
                        extract_file_from_tar(tar_path=file_meta["tar_file"],
                                              file_name=file_meta["file_name"],
                                              dest_folder=folder,
                                              save_file_name=save_file_name)
        image_features_list = []
        meta_features = []

        #     ds[i][0].save(f"{folder}/actual.jpg")
        #     df = pd.DataFrame(metadata.get_indices(indices))
        #     df = df[["url"]]
        #     df["distance"] = dists
        #     df.to_csv(f"{folder}/dup.csv", index=False)
        #     # url = df.url.values[0]
        #     nb += 1
        #     print(nb, len(ds))
        # print(i)


if __name__ == "__main__":
    run(main)
