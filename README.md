rural_urban_biancheng/
├── config.py                 
├── 01_download.py           
├── 02_segment_label.py       
├── 03_train_eval.py          
├── 04_predict.py            
├── 05_report.py              
├── data/
│   ├── raw/
│   │   ├── 边城.txt
│   │   └── 骆驼祥子.txt
│   ├── labeled.csv
│   └── human_overrides.csv  
├── output/
│   ├── metrics.json
│   ├── feature_comparison.csv
│   ├── cm_词频_top200.png
│   ├── cm_词级TFIDF_1_2.png
│   ├── cm_字符TFIDF_2_3.png
│   ├── best_model.joblib
│   ├── report.md
│   └── results.html
├── report.md                             
├── requirements.txt
└── README.md