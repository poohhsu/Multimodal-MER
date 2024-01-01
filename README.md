# DLMAG Final - Multimodal Music Emotion Recognition
This is the final project of Deep Learning for Music Analysis and Generation, which is a multimodal approach that integrates audio, lyric, and symbolic features for music emotion recognition.


## Audio Feature

### Preprocessing
```shell
# Split into training, validation, and testing data
python split_dataset.py --data_dir $1 --save_dir $2
$1: audio data folder (e.g. 'MER31k/')
$2: split path storage folder (e.g. 'data_path/')

# Segment into chunks
python segment.py --data_dir $1 --save_dir $2
$1: split path folder (e.g. 'data_path/')
$2: segmented data storage folder (e.g. 'seg_5/')

# Convert to mel spectrograms
python preprocess.py --data_dir $1 --mel_dir $2
$1: segmented data folder (e.g. 'seg_5/')
$2: mel spectrogram data storage folder (e.g. 'mel_5/')
```

### Training
```shell
# Binary classification
python train_cnn_binary.py --data_dir $1
$1: mel spectrogram data folder (e.g. 'mel_5/')
# or
python train_mert_binary.py --data_dir $1
$1: segmented data folder (e.g. 'seg_5/')
```
```shell
# Multiclass classification
python train_cnn.py --data_dir $1
$1: mel spectrogram data folder (e.g. 'mel_5/')
# or
python train_mert.py --data_dir $1
$1: segmented data folder (e.g. 'seg_5/')
```

### Testing
```shell
# Binary classification
python test_cnn_binary.py --data_dir $1 --save_dir $2
$1: mel spectrogram data folder (e.g. 'mel_5/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict_binary/')
# or
python test_mert_binary.py --data_dir $1 --save_dir $2
$1: segmented data folder (e.g. 'seg_5/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict_binary/')
```
```shell
# Multiclass classification
python test_cnn.py --data_dir $1 --save_dir $2
$1: mel spectrogram data folder (e.g. 'mel_5/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict/')
# or
python test_mert.py --data_dir $1 --save_dir $2
$1: segmented data folder (e.g. 'seg_5/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict/')
```


## Lyric Feature

### Preprocessing
```shell
# Extract vocals
python separate.py --data_dir $1
$1: audio data folder (e.g. 'MER31k/')

# Split into training, validation, and testing data
python split_dataset.py --data_dir $1 --save_dir $2
$1: separated data folder (e.g. 'htdemucs/MER31k/')
$2: split path storage folder (e.g. 'data_path_htdemucs/')

# Extract lyrics
python transcribe.py --data_dir $1 --save_dir $2
$1: split path folder (e.g. 'data_path_htdemucs/')
$2: lyrics storage folder (e.g. 'lyrics_htdemucs/')
```

### GPT
```shell
# Binary classification
python gpt.py --data_dir $1 --save_dir $2 --mode 0
$1: lyrics folder (e.g. 'lyrics_htdemucs/test/')
$2: results storage folder (e.g. 'gpt/test/')
```
```shell
# Multiclass classification
python gpt.py --data_dir $1 --save_dir $2 --mode 1
$1: lyrics folder (e.g. 'lyrics_htdemucs/test/')
$2: results storage folder (e.g. 'gpt/test/')
```

### BERT

#### Training
```shell
# Binary classification
python train_bert_binary.py --data_dir $1
$1: lyrics folder (e.g. 'lyrics_htdemucs/')
```
```shell
# Multiclass classification
python train_bert.py --data_dir $1
$1: lyrics folder (e.g. 'lyrics_htdemucs/')
```

#### Testing
```shell
# Binary classification
python test_bert_binary.py --data_dir $1 --save_dir $2
$1: lyrics folder (e.g. 'lyrics_htdemucs/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict_binary/')
```
```shell
# Multiclass classification
python test_bert.py --data_dir $1 --save_dir $2
$1: lyrics folder (e.g. 'lyrics_htdemucs/')
$2: probability prediction storage folder for late fusion (e.g. 'result_dict/')
```


## Late Fusion (Not yet incorporated symbolic feature)
```shell
# Binary classification
python late_fusion_binary.py --data_dir $1
$1: probability prediction folder (e.g. 'result_dict_binary/')
```
```shell
# Multiclass classification
python late_fusion.py --data_dir $1
$1: probability prediction folder (e.g. 'result_dict/')
```