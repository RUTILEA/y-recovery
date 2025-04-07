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

cd /home/user/shinzaki/y-recovery/data/results
mkdir OK_data_detected
sudo chmod -R 777 .
mv OK_data1/detected/* OK_data_detected/
mv OK_data2/detected/* OK_data_detected/
mv OK_data3/detected/* OK_data_detected/
mv OK_data4/detected/* OK_data_detected/
mv OK_data5/detected/* OK_data_detected/
mv OK_data6/detected/* OK_data_detected/
mv OK_data7/detected/* OK_data_detected/
mv OK_data8/detected/* OK_data_detected/
mv OK_data9/detected/* OK_data_detected/
mv OK_data10/detected/* OK_data_detected/
mv OK_data11/detected/* OK_data_detected/
mv OK_data12/detected/* OK_data_detected/
mv OK_data13/detected/* OK_data_detected/
mv OK_data1/results.json OK_data_detected/OK_data1.json
mv OK_data2/results.json OK_data_detected/OK_data2.json
mv OK_data3/results.json OK_data_detected/OK_data3.json
mv OK_data4/results.json OK_data_detected/OK_data4.json
mv OK_data5/results.json OK_data_detected/OK_data5.json
mv OK_data6/results.json OK_data_detected/OK_data6.json
mv OK_data7/results.json OK_data_detected/OK_data7.json
mv OK_data8/results.json OK_data_detected/OK_data8.json
mv OK_data9/results.json OK_data_detected/OK_data9.json
mv OK_data10/results.json OK_data_detected/OK_data10.json
mv OK_data11/results.json OK_data_detected/OK_data11.json
mv OK_data12/results.json OK_data_detected/OK_data12.json
mv OK_data13/results.json OK_data_detected/OK_data13.json
zip -r OK_data_detected.zip OK_data_detected

mkdir NG_dataA_detected NG_dataB_detected
sudo chmod -R 777 .
mv NG_data_A1/detected/* NG_dataA_detected/
mv NG_data_A2/detected/* NG_dataA_detected/
mv NG_data_A3/detected/* NG_dataA_detected/
mv NG_data_A4/detected/* NG_dataA_detected/
mv NG_data_A5/detected/* NG_dataA_detected/
mv NG_data_A6/detected/* NG_dataA_detected/
mv NG_data_B1/detected/* NG_dataB_detected/
mv NG_data_B2/detected/* NG_dataB_detected/
mv NG_data_B3/detected/* NG_dataB_detected/
mv NG_data_B4/detected/* NG_dataB_detected/
mv NG_data_B5/detected/* NG_dataB_detected/
mv NG_data_B6/detected/* NG_dataB_detected/
mv NG_data_A1/results.json NG_dataA_detected/NG_data_A1.json
mv NG_data_A2/results.json NG_dataA_detected/NG_data_A2.json
mv NG_data_A3/results.json NG_dataA_detected/NG_data_A3.json
mv NG_data_A4/results.json NG_dataA_detected/NG_data_A4.json
mv NG_data_A5/results.json NG_dataA_detected/NG_data_A5.json
mv NG_data_A6/results.json NG_dataA_detected/NG_data_A6.json
mv NG_data_B1/results.json NG_dataB_detected/NG_data_B1.json
mv NG_data_B2/results.json NG_dataB_detected/NG_data_B2.json
mv NG_data_B3/results.json NG_dataB_detected/NG_data_B3.json
mv NG_data_B4/results.json NG_dataB_detected/NG_data_B4.json
mv NG_data_B5/results.json NG_dataB_detected/NG_data_B5.json
mv NG_data_B6/results.json NG_dataB_detected/NG_data_B6.json
zip -r NG_dataA_detected.zip NG_dataA_detected
zip -r NG_dataB_detected.zip NG_dataB_detected