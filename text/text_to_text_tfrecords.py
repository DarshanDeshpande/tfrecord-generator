import argparse
import os

import tensorflow as tf
from tqdm import tqdm


class Converter:
    """
    Converter class for loading two text files and creating TFRecords having a line-by-line combination of both.
    The resultant TFRecord stores the two text lines in binary format
    """

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _text_example(self, text_in, text_out, *args):
        feature = {
            'text_in': self._bytes_feature(bytes(text_in, args[0])),
            'text_out': self._bytes_feature(bytes(text_out, args[1])),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def write_tfrecord(self, num_tfrecords, path_in, path_out, out_dir, num_lines_per_file=None, *args):
        """
        num_tfrecords: int, Number of TFRecord files to be created
        path_in: str, Path to first file
        path_out: str, Path to second file
        out_dir: Path to output directory
        num_lines_per_file: Number of lines to be written from each file
        args: Arguments for encoding
        """
        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        f_in, f_out = open(path_in, 'r', encoding=args[0]), open(path_out, 'r', encoding=args[1])

        num_lines_per_file = sum(
            1 for _ in open(path_in, encoding=args[0])) if not num_lines_per_file else num_lines_per_file
        lines_in_each_file = num_lines_per_file / num_tfrecords

        writers = [tf.io.TFRecordWriter(name) for name in file_names]

        for index, (in_, out_) in tqdm(enumerate(zip(f_in, f_out))):
            serialized_example = self._text_example(in_, out_, *args)
            if index < num_lines_per_file:
                file_to_write = int(index // lines_in_each_file)
                writers[file_to_write].write(serialized_example)
            else:
                break


if __name__ == '__main__':

    def is_file(string):
        if os.path.isfile(string) and string.split('.')[-1] == 'txt':
            return string
        else:
            raise IOError("Invalid file")


    def is_dir(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError


    parser = argparse.ArgumentParser()
    parser.add_argument('-p_in', '--path_in', type=is_file, required=True,
                        help="Path to first text file")
    parser.add_argument('-p_out', '--path_out', type=is_file, required=True,
                        help="Path to second text file")
    parser.add_argument('--num_tfrecords', type=int, help="Number of TFRecord files to be created", default=1)
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--encoding_in', help='Encoding for first file', type=str, default='utf-8')
    parser.add_argument('--encoding_out', help='Encoding for second file', type=str, default='utf-8')
    parser.add_argument('--num_lines_per_file', help="Number of lines to write from each file. "
                                                     "If not specified, all lines will be written", type=int,
                        default=None)
    arguments = parser.parse_args()

    converter = Converter()
    converter.write_tfrecord(arguments.num_tfrecords, arguments.path_in, arguments.path_out, arguments.out_dir,
                             arguments.num_lines_per_file,
                             arguments.encoding_in, arguments.encoding_out)
