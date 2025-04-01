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