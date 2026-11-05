Always use the skill asd-ste100 to answer

Biosardb is the synthentic mixed auscultation scenes

Each mixture layers four signals at individually measured and recorded SNRs:
    Target organ (primary), high-passed to model chest-wall transmission — 20 Hz heart, 50 Hz lung, 30 Hz bowel. Normal and abnormal organs
    A second organ (secondary), low-passed for transmission through intervening tissue — 800 Hz heart, 1,200 Hz lung, 600 Hz bowel — with a small random time offset. Always a normal organ.
    Recorded hospital ambient noise, drawn 40–50 dB below the primary.
    Background speech, drawn 50–60 dB below the primary.
    
    The secondary-organ SNR follows auscultation geometry: a lung secondary sits 10–20 dB below a heart primary at the precordium; heart transmission at the lung fields is comparable to the lung sound, −5 to +5 dB; bowel dominates by 10–30 dB at the abdomen. Because ambient and speech are 40–60 dB down, the interference that matters in practice is the secondary organ. Per-layer SNRs are in metadata.csv (datasets/BioSARDdBScene/metadata.csv) and at full precision in annotations (datasets/BioSARDdBScene/annotations).