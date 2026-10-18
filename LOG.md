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
        
    normal/abnormal heart classification
        edgemamba 10k:
            biosar-heartonly: loss=0.5276 acc=0.674 f1=0.620 auroc=0.778
        tinymamba 48k:
            biosar-heartonly: loss=0.5108 acc=0.678 f1=0.617 auroc=0.782
        separable cnn 13k:
            biosar-heartonly: loss=0.5004 acc=0.686 f1=0.636 auroc=0.803
        simple cnn 97k:
            model input: (batch=32, 1, 1, 64, mel=157)
            Device: cuda; workers: 4; train: 2496; val: 312; test: 312
            
            biosar: 
            biosar-heartonly: loss=0.5060 acc=0.696 f1=0.621 auroc=0.791
            biosarcirco: loss=0.6687 acc=0.571 f1=0.565 auroc=0.607
            circo: loss=0.6100 acc=0.651 f1=0.615 auroc=0.707
            train circo, test biosar: loss=0.6417 acc=0.622 f1=0.459 auroc=0.656
            train biosar, test circo: loss=0.6800 acc=0.609 f1=0.483 auroc=0.606







leave for later scale
dual gpu t4 kaggle: 30% faster time 





Terminology
    Recording: whole audio file
    Clip: window, usually 5s


Preprocess
    Recordings from BioSARdB
    Mono, resample to 4kHz
    Remove DC
    Split into 5s-clip

Training
    The dataset reads each recording.
    It divides the recording into 5-second clips.
    The training loader sends clips in batches.
    The model predicts a score for each clip.
    The training code compares each score with the clip’s label.
    The optimizer updates the model after each batch.
Validation 
    The loader sends one recording at a time.
    A recording can contain several 5-second clips.
    The model predicts a score for each clip.
    The code averages these scores.
    It compares the average score with the recording’s label.
    It uses this result to select the best model checkpoint.

Testing
    Testing uses the same process as validation.
    It loads the best checkpoint.
    It predicts each recording from its clips.
    It reports the final loss and accuracy
