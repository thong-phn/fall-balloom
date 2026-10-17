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
    - Testing uses the same process as validation.
    - It loads the best checkpoint.
    - It predicts each recording from its clips.
    - It reports the final loss and accuracy
