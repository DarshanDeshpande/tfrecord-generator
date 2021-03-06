import argparse
import os
from multiprocessing import Process

import tensorflow as tf
import tensorflow_io as tfio
from tqdm import tqdm


class Converter:
    """
    Converter class for scanning input directory for classes and automatic conversion to TFRecords.
    The resultant TFRecord stores the length, channels, associated label and the raw audio in binary format
    """

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def audio_example(self, path, label):
        audio = tf.io.read_file(path)
        audio_shape = tfio.audio.AudioIOTensor(path).shape

        feature = {
            'length': self._int64_feature(audio_shape[0]),
            'channels': self._int64_feature(audio_shape[1]),
            'label': self._bytes_feature(label) if isinstance(label.numpy(), bytes) else self._int64_feature(label),
            'audio_raw': self._bytes_feature(audio),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @staticmethod
    def _get_paths(directory):
        sub_dirs = tf.io.gfile.glob(f'{directory}/*')
        class_names = [os.path.basename(i) for i in sub_dirs]

        audio_list, label_list = [], []
        for class_num, i in enumerate(sub_dirs):
            audios = tf.io.gfile.glob(f'{i}/*')
            labels = [class_names[class_num]] * (len(audios))
            audio_list.extend(audios)
            label_list.extend(labels)

        return tf.constant(audio_list), tf.constant(label_list)

    def _writer(self, index, file_name, num_audios_per_file, audios, labels):
        with tf.io.TFRecordWriter(file_name) as writer:
            index_start, index_stop = index * num_audios_per_file, (index + 1) * num_audios_per_file

            # If another batch is possible then process normally otherwise process till the end of the list
            if index_stop + num_audios_per_file <= len(audios):
                for string, label in tqdm(
                        zip(audios[index_start:index_stop], labels[index_start:index_stop])):
                    writer.write(self.audio_example(string, label))

            else:
                for string, label in tqdm(zip(audios[index_start:], labels[index_start:])):
                    writer.write(self.audio_example(string, label))

    def write_tfrecord(self, num_tfrecords, directory_path, out_dir):
        """
        This function requires a path to a directory with multiple
        subdirectories having audios arranged in classes.
        The directories should be in the form of

        External dir (Path will be given for this)
          |-- dir A
            |-- file 1
            |-- file 2
          |-- dir B
            |-- file 1
            |-- file 2

        num_tfrecords: Number of TFRecord files to be created. Defaults to 1
        directory_path: Path to External (outermost) directory
        out_dir: Output directory
        """

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        audios, labels = self._get_paths(directory_path)

        num_audios_per_file = len(audios) // len(file_names)

        for index, file_name in enumerate(file_names):
            self._writer(index, file_name, num_audios_per_file, audios, labels)

        print(f"Finished writing {len(audios)} audio files")

    def write_parallely(self, num_tfrecords, directory_path, out_dir):
        """
        This function requires a path to a directory with multiple
        subdirectories having audios arranged in classes.
        The directories should be in the form of

        External dir (Path will be given for this)
          |-- dir A
            |-- file 1
            |-- file 2
          |-- dir B
            |-- file 1
            |-- file 2

        num_tfrecords: Number of TFRecord files to be created. Defaults to 1
        directory_path: Path to External (outermost) directory
        out_dir: Output directory
        """

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        audios, labels = self._get_paths(directory_path)

        num_audios_per_file = len(audios) // len(file_names)

        processes = [Process(target=self._writer, args=(i, j, num_audios_per_file, audios, labels)) for i, j in
                     enumerate(file_names)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print(f"Finished writing {len(audios)} audio files")


if __name__ == '__main__':

    def is_dir(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=is_dir, required=True,
                        help="Path to directory containing image directories")
    parser.add_argument('--num_tfrecords', type=int, help="Number of TFRecord files to be created", default=1)
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--run_parallely', dest='run_parallely', help="Use multi-processing for operations",
                        action='store_true')
    args = parser.parse_args()

    converter = Converter()
    if args.run_parallely:
        converter.write_parallely(args.num_tfrecords, args.path, args.out_dir)
    else:
        converter.write_tfrecord(args.num_tfrecords, args.path, args.out_dir)
