import argparse
import csv
import os
from ast import literal_eval

import tensorflow as tf
from tqdm import tqdm


class CsvConverter:
    """
    Converter class for reading a CSV and automatically converting it's contents to TFRecords.
    The resultant TFRecord stores the associated label and the raw text in binary format
    """

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value, 'utf-8')]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

    @staticmethod
    def _get_type(input_data):
        try:
            return type(literal_eval(input_data))
        except (ValueError, SyntaxError):
            return str

    def _reader(self, csv_path, out_dir):
        filename = f'{out_dir}/' + csv_path.split('/')[-1].split('.')[0] + '.tfrecord' if out_dir else \
            csv_path.split('/')[-1].split('.')[0] + '.tfrecord'

        csv_iterator = iter(csv.reader(open(csv_path)))

        col_names = next(csv_iterator)
        python_dtypes = [self._get_type(col) for col in next(csv_iterator)]
        return csv_iterator, filename, col_names, python_dtypes

    def _csv_example(self, row, col_names, dtypes):
        map_dict = {type(1): self._int64_feature,
                    type(1.5): self._float_feature,
                    type('string'): self._bytes_feature,
                    type(True): self._bytes_feature}

        feature = {}
        for i in range(len(row)):
            feature[col_names[i]] = map_dict[dtypes[i]](row[i])

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def writer(self, csv_file, num_tfrecords, out_dir, num_rows_to_read=None):
        """
        csv_file: str, Path to CSV file
        num_tfrecords: int, Number of TFRecords to be created
        out_dir: str, Path to output directory
        num_rows_to_read: int, Number of rows to read from CSV
        """
        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        total_rows = sum(1 for _ in open(csv_file)) if not num_rows_to_read else num_rows_to_read
        num_rows_per_file = total_rows / num_tfrecords
        writers = [tf.io.TFRecordWriter(f) for f in file_names]

        iterator, file_name, col_names, dtypes = self._reader(csv_file, out_dir)
        for index, row in tqdm(enumerate(iterator)):
            if index >= total_rows:
                break
            file_to_write = int(index // num_rows_per_file)
            writers[file_to_write].write(self._csv_example(row, col_names, dtypes))


if __name__ == '__main__':

    def is_file(string):
        if os.path.isfile(string):
            return string
        else:
            raise IOError(string)


    def is_dir(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=is_file, required=True,
                        help="Path to CSV file")
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--num_tfrecords', type=int, help="Number of TFRecord files to be created", default=1)
    parser.add_argument('--num_rows_to_read', help="Number of lines to write from each file. "
                                                   "If not specified, all lines will be written", type=int,
                        default=None)
    args = parser.parse_args()

    CsvConverter().writer(args.path, args.num_tfrecords, args.out_dir, args.num_rows_to_read)
