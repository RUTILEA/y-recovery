# GSユアサバッテリ検査 PoC2 推論のみ

## 初期設定
### Weightsのダウンロード
```
gdown "https://drive.google.com/u/0/uc?id=1VvlEyD7lyavBTvGO82_zNStANNuF5orX&confirm=t"
```
### Dockerコンテナを起動
```
docker compose up --build -d
```

## 推論
### セルの検査

#### フォルダ構造
`input_dir`（dataフォルダ）に検査するファイルを配置  
なお、フォルダ構成は以下のとおりとします。
```
.
└── data
    ├── OK_data
    │   ├── 1GP200101A0101_正極_20200101_111111
    │   ...
    │   └── 1GP200202A0202_負極_20200202_222222
    └── NG_data
        ├── 1GP200303A0303_正極_20200303_333333
        ...
        └── 1GP200404A0404_負極_20200404_444444
```
`inference_main.py`ファイルのパスを編集して、下記を実行
```
python3 inference_main.py

```

### Inference Pipeline with Defect Visualization (`substance` Defects)

This pipeline detect and visualize defects in images across different imaging axes (`Z-axis`, `Oblique1`, and `Oblique2`). The pipeline provides both **conditional** and **unconditional** inspection modes, along with detailed visualizations including confidence scores.

---

#### 1. Mounting the Dataset Volume

Before running the pipeline, ensure your dataset is properly mounted as a volume in the Docker container.

For example, if your dataset is located at:

```
/media/george/SSD-PUTA/y-recovery/data
```

Mount it in the `docker-compose.yml` like this:

```yaml
services:
  backend:
    image: ubuntu:22.04
    build:
      context: .
      dockerfile: Dockerfile
    container_name: yuasa-server
    volumes:
      - .:/workspace
      - /media/george/SSD-PUTA/y-recovery/data:/workspace/data
      - ./results:/workspace/data/results
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    stdin_open: true
    tty: true
```

---

####  2. Building the Docker Container

To build and start the container:

```bash
docker compose up --build -d --remove-orphans
```

Or simply use:

```bash
make build
```

---

####  3. Running the Inference and Visualization Pipeline

The pipeline can be executed in two modes:

---

##### A. Conditional Inspection Pipeline

**Flow:**  
`Z-axis → Oblique2 → Oblique1`

**Logic:**  
- **Start with Z-axis:**  
  Detect defects. If no defects are found, skip further checks for that image.
- **If defects are found:**  
  - Save the visualization of this image and proceed to **Oblique2** to validate Z-axis detections.  
  - A defect is validated if the center of a bounding box in Oblique2 is within 20 pixels of the Z-axis detection.
  - If validated in Oblique2, stop inspection and save visualization.
- **If not validated in Oblique2:**  
  - Proceed to **Oblique1** with the same 20-pixel threshold.
  - Save visualizations if a match is found.

**Run command:**

```bash
make run
```

Or:

```bash
docker compose exec backend python3 dependent_inspection_with_bounding_boxes.py
```

---

##### B. Unconditional Inspection Pipeline (All-Axis Concurrent Inspection)

In this mode, all three axes are inspected independently and visualized in parallel, regardless of whether a defect is found in Z-axis.

**Run command:**

```bash
make run-all
```

Or:

```bash
docker compose exec backend python3 independent_inspections_with_bounding_boxes.py
```

---

#### Output

Visualized results are saved under the `results` directory, organized by axis:

```plaintext
results/
  ├── OK_data/
  │   ├── oblique1/
  │   ├── oblique2/
  │   └── Zaxis/
  └── results.json
```

---

#### Relevant Scripts

- `dependent_inspection_with_bounding_boxes.py` – Main logic for conditional inspection.
- `independent_inspections_with_bounding_boxes.py` – All-axis inspection script.
- Utilities:
  - `utilities/inference_Zaxis.py`
  - `utilities/inference_ob1.py`
  - `utilities/inference_ob2.py`
  - `utilities/config.py`

For more usage examples and maintenance commands, refer to the included `Makefile`.
