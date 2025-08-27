# SpaceV 1B

SpaceV, initially published by Microsoft, is arguably the best dataset for large-scale Vector Search benchmarks.
It's large enough to stress-test indexing engines running across hundreds of CPU or GPU cores, significantly larger than the traditional [Big-ANN](https://big-ann-benchmarks.com/), which generally operates on just 10 million vectors.
It provides vectors in an 8-bit integral form, empirically optimal for large-scale Information Retrieval and Recommender Systems, capable of leveraging hardware-accelerated quantized dot-products and other SIMD assembly extensions from AVX-512VNNI on x86 and SVE2 on Arm.

The [original dataset](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) was fragmented into 4 GB, which required additional preprocessing before it could be used.
This adaptation re-distributes it under the same [O-UDA license](https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/LICENSE), but in a more accessible format, and augmented with more metadata.
The project description is hosted on [GitHub](https://github.com/ashvardanian/SpaceV) under `ashvardanian/SpaceV`.
The primary merged dataset is hosted on [AWS S3](https://bigger-ann.s3.amazonaws.com/) under `s3://bigger-ann/spacev-1b/`.
The smaller subsample is hosted on [HuggingFace](https://huggingface.co/datasets/unum-cloud/ann-spacev-100m) under `unum-cloud/ann-spacev-100m`.

## Structure

All files are binary matrices in row-major order, prepended by two 32-bit unsigned integers - the number of rows and columns.

- `base.1B.i8bin` - 1.4e9 vectors, each as a 100x 8-bit signed integers. (131 GB)
- `query.30K.i8bin` - 3e4 search queries vectors, each as a 100x 8-bit signed integers. (3 MB)
- `groundtruth.30K.i32bin` - 3e4 ground truth outputs, as a 100x 32-bit integer row IDs. (12 MB)
- `groundtruth.30K.f32bin` - Euclidean distances to each of the 3e4 by 100x search results. (12 MB)

A smaller 100M subset:

- `base.100M.i8bin` - 1e8 vectors subset, each as a 100x 8-bit signed integers. (9 GB)
- `ids.100M.i32bin` - 1e8 vector IDs subset, each as a 100x 32-bit integer row IDs. (380 MB)

## Access

The full dataset is stored on AWS S3, as individual files exceed the limitations of GitHub LFS and Hugging Face Datasets platform.

```bash
$ user@host$ aws s3 ls s3://bigger-ann/spacev-1b/

> YYYY-MM-dd HH:mm:ss 140202072008 base.1B.i8bin
> YYYY-MM-dd HH:mm:ss     11726408 groundtruth.30K.f32bin
> YYYY-MM-dd HH:mm:ss     11726408 groundtruth.30K.i32bin
> YYYY-MM-dd HH:mm:ss      2931608 query.30K.i8bin
```

To download the dataset into a local directory, use the following command:

```bash
mkdir -p datasets/spacev-1b/
aws s3 cp s3://bigger-ann/spacev-1b/ datasets/spacev-1b/ --recursive
```

For convenience, a smaller 100M subset is also available on HuggingFace via LFS:

```bash
mkdir -p datasets/spacev-100m/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/ids.100m.i32bin -P datasets/spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100m.i8bin -P datasets/spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/spacev-100m/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.f32bin -P datasets/spacev-100m/
```

## Usage

The dataset can be loaded with the following Python code, "viewing" the data to avoid pulling everything into memory:

```python
from usearch.io import load_matrix

base_view = load_matrix("base.1B.i8bin", dtype=np.int8, view=True)
queries = load_matrix("query.30K.i8bin", dtype=np.int8)
matches = load_matrix("groundtruth.30K.i32bin", dtype=np.int32)
distances = load_matrix("groundtruth.30K.f32bin", dtype=np.float32)
```

To construct an index and check the recall against ground truth:

```python
from usearch.index import Index, BatchMatches

index = Index(ndim=100, metric="l2sq", dtype="i8")
index.add(None, base_view) # Use incremental keys from 0 to len(base_view)
matches: BatchMatches = index.search(queries)
```

On a modern high-core-count system, constructing the index can be performed at 150'000 vectors per second and will take around 3 hours.
To switch to a smaller dataset, replace the file paths with the corresponding 100M versions:

```python
from usearch.io import load_matrix

base = load_matrix("base.100M.i8bin", dtype=np.int8)
ids = load_matrix("ids.100M.i32bin", dtype=np.int32)
queries = load_matrix("query.30K.i8bin", dtype=np.int8)
matches = load_matrix("groundtruth.30K.i32bin", dtype=np.int32)
distances = load_matrix("groundtruth.30K.f32bin", dtype=np.float32)

from usearch.index import Index, BatchMatches

index = Index(ndim=100, metric="l2sq", dtype="i8")
index.add(ids, base)
matches: BatchMatches = index.search(queries)
```
