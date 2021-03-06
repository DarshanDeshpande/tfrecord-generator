import os
import argparse
import tensorflow as tf
import sqlite3
from tqdm import tqdm
import re


class SQLiteConverter:
    """
    Converter class for converting SQLite table contents to TFRecords
    """
    def __init__(self):
        self.sqlite_dtypes = [
            ['INT', 'INTEGER', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT', 'UNSIGNED', 'INT2', 'INT8'],
            ['CHARACTER', 'VARCHAR', 'VARYING CHARACTER', 'NCHAR', 'NATIVE CHARACTER', 'NVARCHAR', 'TEXT', 'CLOB'],
            ['REAL', 'DOUBLE', 'DOUBLE PRECISION', 'FLOAT'],
            ['NUMERIC', 'DECIMAL', 'BOOLEAN', 'DATE', 'DATETIME']]

    @staticmethod
    def _get_table_info(cursor, table_name, col_list):
        cursor.execute(f'PRAGMA table_info({table_name})')
        result = cursor.fetchall()
        col_names = [i[1] for i in result if i[1] in col_list] if col_list else [i[1] for i in result]
        col_dtypes = [i[2] for i in result if i[1] in col_names]
        return col_names, col_dtypes

    def _find_dtype(self, col_dtypes):
        py_dtypes = [tf.int32, tf.string, tf.float32, tf.string]
        converted_list = []

        col_dtypes = [re.sub('\([^)]*\)', '', i) for i in col_dtypes]
        for c_dtype in col_dtypes:
            for i, dtype_list in enumerate(self.sqlite_dtypes):
                if c_dtype in dtype_list:
                    converted_list.append(py_dtypes[i])

        return converted_list

    def _create_feature_dict(self, py_dtypes, col_names, data, writer):
        d, null_count = {}, 0
        mappings = {tf.string: self._bytes_feature,
                    tf.int32: self._int64_feature,
                    tf.float32: self._float_feature}

        for entry in data:
            if None not in entry:
                for index, (feature, name) in enumerate(zip(py_dtypes, col_names)):
                    d[name] = mappings[feature](entry[index])
                serialized = tf.train.Example(features=tf.train.Features(feature=d)).SerializeToString()
                writer.write(serialized)
            else:
                entry = ['' if x is None else x for x in entry]
                for index, (feature, name) in enumerate(zip(py_dtypes, col_names)):
                    d[name] = mappings[feature](entry[index])
                serialized = tf.train.Example(features=tf.train.Features(feature=d)).SerializeToString()
                writer.write(serialized)

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        else:
            value = value.encode()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def write_to_file(self, db_path, table_name, out_dir, col_list=None,
                      buffer_enable=True, buffer_size=500, n_entries=None):
        """
        db_path: str, Path to SQLite Database
        table_name: str, Name of required table associated with the database
        out_dir: str, Path to output directory
        col_list: str, List of columns
        buffer_enable: bool, Whether to enable buffering or not
        buffer_size: int, Buffer size if buffer is enabled
        n_entries: int, Number of rows to be fetched
        """
        _path = os.path.join(out_dir, f"{table_name}.tfrecord")
        if n_entries and n_entries <= buffer_size:
            buffer_size = n_entries
            buffer_enable = False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        col_names, col_dtypes = self._get_table_info(cursor, table_name, col_list)
        col_dtypes = self._find_dtype(col_dtypes)

        with tf.io.TFRecordWriter(_path) as writer:
            if n_entries:
                if buffer_enable:
                    steps = n_entries // buffer_size
                    rem = n_entries % buffer_size

                    for i in tqdm(range(steps + 1)):
                        if (i + 1) * buffer_size <= n_entries:
                            cursor.execute(f"SELECT {','.join(col_names)} FROM {table_name} LIMIT {i * buffer_size}, {buffer_size}")
                            self._create_feature_dict(col_dtypes, col_names, cursor.fetchall(), writer)

                        elif (i + 1) * buffer_size > n_entries and rem != 0:
                            cursor.execute(
                                f"SELECT {','.join(col_names)} FROM {table_name} LIMIT {i * buffer_size}, {n_entries - (i * buffer_size)}")
                            self._create_feature_dict(col_dtypes, col_names, cursor.fetchall(), writer)
                else:
                    cursor.execute(f"SELECT {','.join(col_names)} FROM {table_name} LIMIT {n_entries}")
                    self._create_feature_dict(col_dtypes, col_names, cursor.fetchall(), writer)
            else:
                cursor.execute(f"SELECT {','.join(col_names)} FROM {table_name}")
                self._create_feature_dict(col_dtypes, col_names, cursor.fetchall(), writer)
        print(f"Written to {_path}")


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
                        help="Path to SQLite file")
    parser.add_argument('--table_name', type=str, help="Name of the SQLite table", required=True)
    parser.add_argument('--n_entries', type=int, help="[Optional] Number of rows to be extracted from the table")
    parser.add_argument('--col_list', type=str, help="Comma separated names of columns to be extracted", required=True)
    parser.add_argument('--out_dir', type=is_dir, help="Path for directory where TFRecord files will be stored")
    parser.add_argument('--buffer_enable', dest='buffer_enable',
                        help="Enable buffering. Recommended for large files",
                        action='store_true')
    parser.add_argument('--buffer_size', type=int, help="Buffer size. Only applicable if --buffer_enable is active. "
                                                        "Default=500", default=500)

    args = parser.parse_args()
    path = args.path.replace('\\', '/')

    columns = [str(item) for item in args.col_list.split(',')] if args.col_list else None
    SQLiteConverter().write_to_file(path, args.table_name, args.out_dir, columns,
                                    args.buffer_enable, args.buffer_size, args.n_entries)
