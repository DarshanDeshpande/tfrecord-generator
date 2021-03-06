import argparse
import os
from multiprocessing import Process

import tensorflow as tf
from tqdm import tqdm


class Converter:
    '''
    Converter class for scanning input directory for classes and automatic conversion to TFRecords.
    The resultant TFRecord stores the height, width, channels, associated label (inferred from directory) and the raw image in binary format
    '''

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _image_example(self, path, label, _resize=None, maintain_aspect_ratio=False, _grayscale=False):
        image = tf.io.read_file(path)
        loaded_image = tf.image.decode_image(image, channels=3)
        if _resize is not None:
            loaded_image = tf.image.resize(loaded_image, _resize, preserve_aspect_ratio=maintain_aspect_ratio)
        if _grayscale:
            loaded_image = tf.image.rgb_to_grayscale(loaded_image)
        image_shape = loaded_image.shape

        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'channels': self._int64_feature(image_shape[2]),
            'label': self._bytes_feature(label) if isinstance(label.numpy(), bytes) else self._int64_feature(label),
            'image_raw': self._bytes_feature(tf.io.encode_jpeg(tf.cast(loaded_image, tf.uint8))),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @staticmethod
    def _get_paths(directory):
        sub_dirs = tf.io.gfile.glob(f'{directory}/*')
        class_names = [os.path.basename(i) for i in sub_dirs]

        image_list, label_list = [], []
        for class_num, i in enumerate(sub_dirs):
            images = tf.io.gfile.glob(f'{i}/*')
            labels = [class_names[class_num]] * (len(images)) if class_names else [class_num] * (len(images))
            image_list.extend(images)
            label_list.extend(labels)

        return tf.constant(image_list), tf.constant(label_list)

    def _writer(self, index, file_name, num_images_per_file, images, labels, *args):
        with tf.io.TFRecordWriter(file_name) as writer:
            index_start, index_stop = index * num_images_per_file, (index + 1) * num_images_per_file

            # If another batch is possible then process normally otherwise process till the end of the list
            if index_stop + num_images_per_file <= len(images):
                for image_string, label in tqdm(
                        zip(images[index_start:index_stop], labels[index_start:index_stop])):
                    writer.write(self._image_example(image_string, label, *args))

            else:
                for image_string, label in tqdm(zip(images[index_start:], labels[index_start:])):
                    writer.write(self._image_example(image_string, label, *args))

    def write_tfrecord(self, num_tfrecords, directory_path, out_dir, *args):
        '''
        This function requires a path to a directory with multiple
        subdirectories having images arranged in classes.
        The directories should be in the form of

        External dir
          |-- dir A
            |-- img 1
            |-- img 2
          |-- dir B
            |-- img 1
            |-- img 2

        file_names: List or iterable
        directory_path: Path to External (outermost) directory
        class_names: (Optional) List or tuple with names of classes.
                      Length should be equal to number of sub-directories
        args: Arguments for augmentation
        '''

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        images, labels = self._get_paths(directory_path)

        num_images_per_file = len(images) // len(file_names)

        for index, file_name in enumerate(file_names):
            self._writer(index, file_name, num_images_per_file, images, labels, *args)

        print(f"Finished writing {len(images)} images")

    def write_parallely(self, num_tfrecords, directory_path, out_dir, *args):
        '''
        This function requires a path to a directory with multiple
        subdirectories having images arranged in classes.
        The directories should be in the form of

        External dir
          |-- dir A
            |-- img 1
            |-- img 2
          |-- dir B
            |-- img 1
            |-- img 2

        file_names: List or iterable
        directory_path: Path to External (outermost) directory
        class_names: (Optional) List or tuple with names of classes.
                      Length should be equal to number of sub-directories
        args: Arguments for augmentation
        '''

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        images, labels = self._get_paths(directory_path)

        num_images_per_file = len(images) // len(file_names)
        processes = [Process(target=self._writer,
                             args=(i, j, num_images_per_file, images, labels, *args)) for i, j
                     in
                     enumerate(file_names)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print(f"Finished writing {len(images)} images")


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
    parser.add_argument('--resize',
                        help='Resize list input in the form of `height, width`. Example: --resize 300,400', type=str)
    parser.add_argument('--maintain_aspect_ratio', dest='maintain_aspect_ratio', help="Maintains aspect ratio",
                        action='store_true')
    parser.add_argument('--grayscale', dest='grayscale', help="Maintains aspect ratio",
                        action='store_true')
    arguments = parser.parse_args()

    resize = tf.constant([int(item) for item in arguments.resize.split(',')], tf.int32)
    converter = Converter()

    if arguments.run_parallely:
        converter.write_parallely(arguments.num_tfrecords, arguments.path, arguments.out_dir, resize,
                                  arguments.maintain_aspect_ratio, arguments.grayscale)
    else:
        converter.write_tfrecord(arguments.num_tfrecords, arguments.path, arguments.out_dir, resize,
                                 arguments.maintain_aspect_ratio, arguments.grayscale)
