# GSユアサバッテリ検査 PoC2 

## セットアップ
### Dockerイメージのビルド
```
docker build -t <コンテナ名> .
```
### Dockerコンテナを起動<br>
コマンド例）

```
docker run --rm -it --gpus all -v "D:\development_data:/workspace/data" -v "%CD%:/workspace" detectron2-container /bin/bash
```

## 推論
### セルの検査

#### フォルダ構造
`input_dir`に検査したいフォルダを配置
```
.
└── data
    └── NG_data
        ├── 1GP170529A0214_正極_20170708_180638
        ...
        └── 1GP170703A0027_負極_20170801_235645
```
`inference_main.py`ファイルのパスを編集して、下記を実行
```
python inference_main.py
```

## 学習
### データセット作成
`labelme_to_coco.py`: labelmeで作成したアノテーションデータをcoco形式に変換<br>
`augument.py`: 良品画像に不良を複数貼り付けて、データの水増しを行う<br>
`crop_bead.py`, `crop_ob1.py`, `crop_ob2.py`: 入力サイズに切り出し<br>

### 実行
`train.py`ファイルのパスを編集して、下記を実行
```
python train.py
```