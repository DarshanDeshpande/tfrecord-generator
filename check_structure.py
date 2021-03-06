import argparse
import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', required=True, help='Path to TFRecord file that needs to be analysed')
    parser.add_argument('--num_visualize', help="Number of samples to visualize", default=1)
    args = parser.parse_args()

    raw_dataset = tf.data.TFRecordDataset(args.tfrecord_path)
    for raw_record in raw_dataset.take(int(args.num_visualize)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
