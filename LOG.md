Exp
single gpu

    all class
        simple cnn 97k: loss=0.9234 acc=0.606 bal_acc=0.544 macro_f1=0.510 macro_auroc=0.875
            biosar: 0.5
                class            n   recall
                Bowel-normal     48   0.896
                Bowel-abnormal   11   0.636
                Heart-normal    427   0.323
                Heart-abnormal  287   0.484
                Joint-normal      5   1.000
                Joint-abnormal    3   0.000
                Lung-normal    1287   0.888
                Lung-abnormal   461   0.124
        separablecnn 13k: loss=1.0068 acc=0.584 bal_acc=0.520 macro_f1=0.489 macro_auroc=0.858
            biosar: 0.5
                class            n   recall
                Bowel-normal     48   1.000
                Bowel-abnormal   11   0.091
                Heart-normal    427   0.445
                Heart-abnormal  287   0.645
                Joint-normal      5   0.800
                Joint-abnormal    3   0.333
                Lung-normal    1287   0.796
                Lung-abnormal   461   0.050
        attn-cnn
        Test: loss=0.9599 acc=0.585 macro_recall=0.488 bal_acc=0.488 macro_specificity=0.905 macro_f1=0.461 macro_auroc=0.858
            class            n   recall  specificity  f1
            Bowel-normal     48   0.958      0.997    0.902
            Bowel-abnormal   11   0.273      0.999    0.353
            Heart-normal    427   0.138      0.980    0.224
            Heart-abnormal  287   0.432      0.946    0.466
            Joint-normal      5   0.400      0.999    0.400
            Joint-abnormal    3   0.667      0.998    0.444
            Lung-normal    1287   0.928      0.346    0.725
            Lung-abnormal   461   0.108      0.972    0.176
        
    normal/abnormal heart from circo classification
        edgemamba 10k:
            biosar-heartonly: loss=0.5276 acc=0.674 f1=0.620 auroc=0.778
        tinymamba 48k:
            biosar-heartonly: loss=0.5108 acc=0.678 f1=0.617 auroc=0.782
        separable cnn 13k:
            biosar-heartonly: loss=0.5004 acc=0.686 f1=0.636 auroc=0.803
        simple cnn 97k:
            biosar-heartonly: loss=0.5060 acc=0.696 f1=0.621 auroc=0.791
            
            biosarcirco: loss=0.6687 acc=0.571 f1=0.565 auroc=0.607
            circo: loss=0.6100 acc=0.651 f1=0.615 auroc=0.707
                train circo, test biosar: loss=0.6417 acc=0.622 f1=0.459 auroc=0.656
                train biosar, test circo: loss=0.6800 acc=0.609 f1=0.483 auroc=0.606
            
            biosarbmd: 
                val_auroc: 0.690104, test_sensitivity: 0.875000, test_specificity: 0.812500, test_f1: 0.913043, test_auroc: 0.915799, test_auprc: 0.978986, test_mcc: 0.612372
            bmd: 
            







