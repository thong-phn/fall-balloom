Exp
single gpu
    normal/abnormal heart classification
        simple cnn
            model input: (batch=32, 1, 1, 64, mel=157)
            Device: cuda; workers: 4; train: 2496; val: 312; test: 312
            biosarcirco: test_loss=0.6511, test_acc=0.577
            circo: test_loss0.6369, test_acc=0.599 


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
