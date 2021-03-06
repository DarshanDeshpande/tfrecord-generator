import argparse
import os
from multiprocessing import Process

import tensorflow as tf
from tqdm import tqdm


class Converter:
    """
    Converter class for scanning two input directories for images and writing them to TFRecord.
    The resultant TFRecord stores the height, width, channels and the raw image in binary format for each pair
    """

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _get_image(path, resize, maintain_aspect_ratio, grayscale):
        image = tf.io.read_file(path)
        loaded_image = tf.image.decode_image(image, channels=3)
        if resize is not None:
            loaded_image = tf.image.resize(loaded_image, size=resize, preserve_aspect_ratio=maintain_aspect_ratio)
        if grayscale:
            loaded_image = tf.image.rgb_to_grayscale(loaded_image)

        image_shape = loaded_image.shape
        loaded_image = tf.io.encode_jpeg(tf.cast(loaded_image, tf.uint8), format="grayscale" if grayscale else "rgb")
        return loaded_image, image_shape

    def _image_example(self, path1, path2, *args):

        image_in, image_shape_in = self._get_image(path1, args[0], args[1], args[2])
        image_out, image_shape_out = self._get_image(path2, args[3], args[4], args[5])

        feature = {
            'height_inp': self._int64_feature(image_shape_in[0]),
            'width_inp': self._int64_feature(image_shape_in[1]),
            'channels_inp': self._int64_feature(image_shape_in[2]),
            'image_raw_inp': self._bytes_feature(image_in),
            'height_out': self._int64_feature(image_shape_out[0]),
            'width_out': self._int64_feature(image_shape_out[1]),
            'channels_out': self._int64_feature(image_shape_out[2]),
            'image_raw_out': self._bytes_feature(image_out),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @staticmethod
    def _get_paths(directory1, directory2):
        image_list = tf.io.gfile.glob(f'{directory1}/*')
        label_list = tf.io.gfile.glob(f'{directory2}/*')

        return tf.constant(image_list), tf.constant(label_list)

    def _writer(self, index, file_name, num_images_per_file, images, labels, *args):
        """
        Helper function for iterating and writing
        """
        with tf.io.TFRecordWriter(file_name) as writer:
            index_start, index_stop = index * num_images_per_file, (index + 1) * num_images_per_file

            # If another batch is possible then process normally otherwise process till the end of the list
            if index_stop + num_images_per_file <= len(images):
                for image1_string, image2_string in tqdm(
                        zip(images[index_start:index_stop], labels[index_start:index_stop])):
                    writer.write(
                        self._image_example(image1_string, image2_string, *args))

            else:
                for image1_string, image2_string in tqdm(zip(images[index_start:], labels[index_start:])):
                    writer.write(
                        self._image_example(image1_string, image2_string, *args))

    def write_tfrecord(self, num_tfrecords, directory_path1, directory_path2, out_dir, *args):
        """
        num_tfrecords: int, Number of TFRecords to create
        directory_path1: str, Path to input images
        directory_path2: str, Path to output images
        out_dir: str, Path to output directory
        args: Miscellaneous arguments for `writer`
        """

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        images, labels = self._get_paths(directory_path1, directory_path2)

        num_images_per_file = len(images) // len(file_names)

        for index, file_name in enumerate(file_names):
            self._writer(index, file_name, num_images_per_file, images, labels, *args)

        print(f"Finished writing {len(images)} images")

    def write_parallely(self, num_tfrecords, directory_path1, directory_path2, out_dir, *args):
        """
        num_tfrecords: int, Number of TFRecords to create
        directory_path1: str, Path to input images
        directory_path2: str, Path to output images
        out_dir: str, Path to output directory
        args: Miscellaneous arguments for `writer`
        """

        file_names = [f"{out_dir}/{i}.tfrecord" if out_dir else f"{i}.tfrecord" for i in range(num_tfrecords)]
        images, labels = self._get_paths(directory_path1, directory_path2)

        num_images_per_file = len(images) // len(file_names)

        processes = [Process(
            target=self._writer,
            args=(i, j, num_images_per_file, images, labels, *args)) for i, j in enumerate(file_names)]
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
    parser.add_argument('-in_image_dir', '--in_images', type=is_dir, required=True,
                        help="Path to directory containing input images")
    parser.add_argument('-out_image_dir', '--out_images', type=is_dir, required=True,
                        help="Path to directory containing output images")
    parser.add_argument('--num_tfrecords', type=int, help="Number of TFRecord files to be created", default=1)
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--run_parallely', dest='run_parallely', help="Use multi-processing for operations",
                        action='store_true')
    parser.add_argument('--resize_in',
                        help='Resize image. Takes in a list input in the form of `height, width`.'
                             ' Example: --resize_in 300,400', type=str)
    parser.add_argument('--resize_out',
                        help='Resize image. Takes in a list input in the form of `height, width`.'
                             'Example: --resize_out 300,400', type=str)
    parser.add_argument('--maintain_aspect_ratio_in', dest='maintain_aspect_ratio_in',
                        help="To maintain aspect ratio or not",
                        action='store_true')
    parser.add_argument('--maintain_aspect_ratio_out', dest='maintain_aspect_ratio_out',
                        help="To maintains aspect ratio or not",
                        action='store_true')
    parser.add_argument('--grayscale_in', dest='grayscale_in',
                        help="To maintain aspect ratio or not",
                        action='store_true')
    parser.add_argument('--grayscale_out', dest='grayscale_out',
                        help="To maintains aspect ratio or not",
                        action='store_true')
    arguments = parser.parse_args()

    resize_in = [int(item) for item in arguments.resize_in.split(',')]
    resize_out = [int(item) for item in arguments.resize_out.split(',')]

    converter = Converter()

    if arguments.run_parallely:
        converter.write_parallely(arguments.num_tfrecords, arguments.in_images, arguments.out_images, arguments.out_dir,
                                  resize_in, arguments.maintain_aspect_ratio_in, arguments.grayscale_in,
                                  resize_out, arguments.maintain_aspect_ratio_out, arguments.grayscale_out)
    else:
        converter.write_tfrecord(arguments.num_tfrecords, arguments.in_images, arguments.out_images, arguments.out_dir,
                                 resize_in, arguments.maintain_aspect_ratio_in, arguments.grayscale_in, resize_out,
                                 arguments.maintain_aspect_ratio_out, arguments.grayscale_out)
