## FGVC8-teamwork
https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/code
# Results
## Semi-iNature
|  Training Data  | Method  | input|Top-1 Val. Accuracy (%)|epoch|
|  ----  | ----  |:----:|:----:|:----:|
| Labeled train images  | ResNet-50 ||31.00|-|
| Labeled train images  | resnet-101 ||34.02|199|
| Labeled train images  | resnet-152  ||39.6|59|
| Labeled train images  | eff_b1 ||36.29|199|
| Labeled train images  | eff_b3b ||42.00|199|
| Labeled train images  | eff_b4b |360x360|44.98|149|
| Labeled train images  | eff_b4b |800x800|46.67|18|
## Label
|images per category|  >=5  | >10  | >20|>30|>40|  >50  | >60  | >70|>80|
|:----:| :----: | :----:|:----:|:----:|:----:| :----:|:----:|:----:|:----:|
|categories| 810 | 329  |126|45|18| 9  |5 |2|0|
## CDP
|kNN(k)|  threshold | images| classes|
|:----:| :----: |:----:| :----: |
|80| 0.5 |147427| 79002 |
|60| 0.5 |126985| 56353 |
|50| 0.5 |114147| 42492 |
|40| 0.5 |98625| 29010 |
