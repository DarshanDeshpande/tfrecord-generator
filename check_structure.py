import argparse
import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', required=True, help='Path to TFRecord file that needs to be analysed')
    args = parser.parse_args()

    raw_dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
