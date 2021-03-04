# TFRecord Generator

<p align="center">
  <a href="">
    <img src="https://i.imgur.com/cxIbm7n.png" alt="Logo"  width="35%">
  </a>
</p>

## Introduction

This repository contains scripts for conversion of data required for most commonly found Machine Learning tasks to TFRecords. TFRecords are versatile storage formats which store serialized data in byte format which can be loaded directly into a Tensorflow pipeline using a `tf.data.TFRecordDataset`. The library supports splitting of data into multiple records and also tries to minimize slow I/O write speeds by running parallel processes which can be enabled by adding the `--run_parallely` parameter to the scripts that support it.

The repository currently supports the following scripts:

<li>Images
    <ul>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/image/image_to_tfrecords.py">Images to TFRecords</a></li>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/image/image_to_image_tfrecords.py">Image to Image TFRecords </a>(Useful for generative models)</li>
    </ul>
</li>
<li>Text
    <ul>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/text/csv_to_tfrecords.py">CSV To TFRecords</a></li>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/text/sqlite_to_tfrecords.py">SQLite To TFRecords</a></li>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/text/text_to_text_tfrecords.py">Text To Text TFRecords</a> (Useful for generative models)</li>
    </ul>
</li>
<li>Audio
    <ul>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/audio/audio_to_tfrecords.py">Audio To TFRecords </a></li>
        <li><a href="https://github.com/DarshanDeshpande/tfrecord-generator/blob/master/audio/spectrogram_to_tfrecords.py">Spectrograms To TFRecords</a></li>
    </ul>
</li>

## Installation

The library has minimal requirements which can be installed by running
```
pip install -r requirements.txt
```

## How to use

Steps to use the library

1. Clone the repository
```
git clone https://github.com/DarshanDeshpande/tfrecord-generator
```

2. Choose your preferred conversion script. See all available options by running
```python
python script_name.py --help
```

For example, if you want to convert your images to TFRecords then through your cmd run the following example
```python
python image_to_tfrecords.py -p "path/to/directory/containing/subdirectories" \
--nfiles 3 \
--out_dir "path/to/output/directory" \
--run_parallely \
--resize 500,500 \
--maintain_aspect_ratio \
--grayscale
```
You can then check the structure of the file created using
```python
python check_structure.py --tfrecord_path "path/to/tfrecord"
```