import argparse
import csv
import os
from ast import literal_eval
from multiprocessing import Process

import tensorflow as tf
from tqdm import tqdm


class Converter:
    """
    Converter class for reading a CSV and automatically converting it's contents to
    TFRecords along with the corresponding images.
    The resultant TFRecord stores the associated data and the raw image in binary format
    """

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        if isinstance(value, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        else:
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

    @staticmethod
    def _reader(csv_file):
        rows = list(csv.reader(open(csv_file, 'r')))
        col_names = rows.pop(0)
        row_dict = {}
        for row in rows:
            row_dict[row[0]] = row[1:]

        return rows, row_dict, col_names

    def _image_example(self, file_name, values, col_names, dir_path):
        map_dict = {type(1): self._int64_feature,
                    type(1.5): self._float_feature,
                    type('string'): self._bytes_feature,
                    type(True): self._bytes_feature}

        feature = {col_names[0]: self._bytes_feature(tf.io.read_file(os.path.join(dir_path, file_name)))}
        for index, value in enumerate(values):
            feature[col_names[index + 1]] = map_dict[self._get_type(value)](value)
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def write(self, tfrecord_name, row_dict, col_names, dir_path):
        with tf.io.TFRecordWriter(tfrecord_name) as writer:
            for index, (file_name, values) in tqdm(enumerate(row_dict.items())):
                writer.write(self._image_example(file_name, values, col_names, dir_path))

    def writer(self, csv_file, dir_path, num_files, out_dir, num_rows_to_read=None):
        """
        csv_file: str, Path to CSV file
        dir_path: str, Path to directory containing images
        num_files: int, Number of TFRecord files to create
        out_dir: str, Path to output directory
        num_rows_to_read: int, Number of rows to read from the CSV file
        """
        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_files)]

        rows, row_dict, col_names = self._reader(csv_file)
        total_rows = len(rows) if not num_rows_to_read else num_rows_to_read

        num_rows_per_file = total_rows / num_files
        writers = [tf.io.TFRecordWriter(f) for f in file_names]

        for index, (file_name, values) in tqdm(enumerate(row_dict.items())):
            if index >= total_rows:
                break
            file_to_write = int(index // num_rows_per_file)
            writers[file_to_write].write(self._image_example(file_name, values, col_names, dir_path))

    def write_parallely(self, csv_file, dir_path, num_files, out_dir, num_rows_to_read=None):
        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_files)]

        rows, row_dict, col_names = self._reader(csv_file)
        total_rows = len(rows) if not num_rows_to_read else num_rows_to_read
        num_rows_per_file = int(total_rows / num_files)

        processes = []
        items = list(row_dict.items())
        for i in range(num_files):
            if i == num_files - 1:
                p = Process(target=self.write,
                            args=(file_names[i], dict(items[num_rows_per_file * i:]), col_names, dir_path))
            else:
                p = Process(target=self.write, args=(
                    file_names[i], dict(items[num_rows_per_file * i: num_rows_per_file * (i + 1)]), col_names,
                    dir_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


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
    parser.add_argument('--csv_path', type=is_file, required=True,
                        help="Path to CSV file")
    parser.add_argument('--images_dir', type=is_dir, help="Path for directory where images are stored")
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--nfiles', type=int, help="Number of TFRecord files to be created", default=1)
    parser.add_argument('--num_rows_to_read', help="Number of rows to write from CSV file. "
                                                   "If not specified, all rows will be written", type=int,
                        default=None)
    parser.add_argument('--run_parallely', dest='run_parallely', help="Use multi-processing for operations",
                        action='store_true')
    args = parser.parse_args()

    if args.run_parallely:
        Converter().write_parallely(args.csv_path, args.images_dir, args.nfiles, args.out_dir, args.num_rows_to_read)

    else:
        Converter().writer(args.csv_path, args.images_dir, args.nfiles, args.out_dir, args.num_rows_to_read)
