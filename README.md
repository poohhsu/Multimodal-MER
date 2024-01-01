# DLMAG Final - Multimodal Music Emotion Recognition
This is the final project of Deep Learning for Music Analysis and Generation, which is a multimodal approach that integrates audio, lyric, and symbolic features for music emotion recognition.


## Audio Feature

### Preprocessing

#### Split into training, validation, and testing data
```
python split_dataset.py --data_dir $1 --save_dir $2
```
where `$1` is the audio data folder (e.g. 'MER31k/'), `$2` is the split path storage folder (e.g. 'data_path/')

#### Segment into chunks
```
python segment.py --data_dir $1 --save_dir $2
```
where `$1` is the split path folder (e.g. 'data_path/'), `$2` is the segmented data storage folder (e.g. 'seg_5/')

#### Convert to mel spectrograms
```
python preprocess.py --data_dir $1 --mel_dir $2
```
where `$1` is the segmented data folder (e.g. 'seg_5/'), `$2` is the mel spectrogram data storage folder (e.g. 'mel_5/')

### Training

#### Binary classification
```
python train_cnn_binary.py --data_dir $1
```
where `$1` is the mel spectrogram data folder (e.g. 'mel_5/'), or
```
python train_mert_binary.py --data_dir $1
```
where `$1` is the segmented data folder (e.g. 'seg_5/')

#### Multiclass classification
```
python train_cnn.py --data_dir $1
```
where `$1` is the mel spectrogram data folder (e.g. 'mel_5/'), or
```
python train_mert.py --data_dir $1
```
where `$1` is the segmented data folder (e.g. 'seg_5/')

### Testing

#### Binary classification
```
python test_cnn_binary.py --data_dir $1 --save_dir $2
```
where `$1` is the mel spectrogram data folder (e.g. 'mel_5/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict_binary/'), or
```
python test_mert_binary.py --data_dir $1 --save_dir $2
```
where `$1` is the segmented data folder (e.g. 'seg_5/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict_binary/')

#### Multiclass classification
```
python test_cnn.py --data_dir $1 --save_dir $2
```
where `$1` is the mel spectrogram data folder (e.g. 'mel_5/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict/'), or
```
python test_mert.py --data_dir $1 --save_dir $2
```
where `$1` is the segmented data folder (e.g. 'seg_5/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict/')


## Lyric Feature

### Preprocessing

#### Extract vocals
```
python separate.py --data_dir $1
```
where `$1` is the audio data folder (e.g. 'MER31k/')

#### Split into training, validation, and testing data
```
python split_dataset.py --data_dir $1 --save_dir $2
```
where `$1` is the separated data folder (e.g. 'htdemucs/MER31k/'), `$2` is the split path storage folder (e.g. 'data_path_htdemucs/')

#### Extract lyrics
```
python transcribe.py --data_dir $1 --save_dir $2
```
where `$1` is the split path folder (e.g. 'data_path_htdemucs/'), `$2` is the lyrics storage folder (e.g. 'lyrics_htdemucs/')

### GPT

#### Binary classification
```
python gpt.py --data_dir $1 --save_dir $2 --mode 0
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/test/'), `$2` is the results storage folder (e.g. 'gpt/test/')

#### Multiclass classification
```
python gpt.py --data_dir $1 --save_dir $2 --mode 1
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/test/'), `$2` is the results storage folder (e.g. 'gpt/test/')

### BERT

#### Training

##### Binary classification
```
python train_bert_binary.py --data_dir $1
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/')

##### Multiclass classification
```
python train_bert.py --data_dir $1
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/')

#### Testing

##### Binary classification
```
python test_bert_binary.py --data_dir $1 --save_dir $2
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict_binary/')

##### Mlticlass classification
```
python test_bert.py --data_dir $1 --save_dir $2
```
where `$1` is the lyrics folder (e.g. 'lyrics_htdemucs/'), `$2` is the probability prediction storage folder for late fusion (e.g. 'result_dict/')


## Late Fusion (Not yet incorporated symbolic feature)

### Binary classification
```
python late_fusion_binary.py --data_dir $1
```
where `$1` is the probability prediction folder (e.g. 'result_dict_binary/')

### Multiclass classification
```
python late_fusion.py --data_dir $1
```
where `$1` is the probability prediction folder (e.g. 'result_dict/')