
# Defines routines for loading and saving dataframes with configuration data.


from genericpath import isfile
import io
import os
import re
import gzip
import shutil
import tarfile
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Optional, Tuple, Iterable, Dict, Any

import numpy as np
import pandas as pd

import tools.configuration as _cfg


# TODO: move somewhere else
class Memoize:
    def __init__(self, func=None, value=None):
        self._func = func if value is None else None
        self._value = value
    def __call__(self):
        if self._func is not None:
            self._value = self._func()
            assert self._value is not None  # `func()` shall not return `None`
            self._func = None
        return self._value
    def is_none(self):
        return self._func is None and self._value is None

class MemoizeResource(Memoize):
    def __init__(self, func=None, value=None):
        super().__init__(func=func, value=value)
    def close(self):
        if self._value is not None:
            self._value.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class DelayRemovedNamedTemporaryFile:
    def __init__(self, mode: str = 'w+b'):
        self._file = tempfile.NamedTemporaryFile(mode=mode, delete=False)
        print("temporary file:", self._file.name)

    @property
    def file(self):
        return self._file

    @property
    def name(self):
        return self._file.name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if not self._file.closed:
                self._file.close()
            os.unlink(self._file.name)
        except FileNotFoundError:
            pass


_header_comment_serialization_options = _cfg.ConfigSerializationOptions(
    short=False,
    section_separator=True,
    prefix='# ', prefix_pattern='^\\s*#\\s*',
    section_format='{}:', section_pattern='(.+?)\\s*:\\s*(?:#\\s*(.*)|)$',                             # "# section:  # comment"
    name_value_format='  - {}: {}', name_value_pattern='-\\s*(.+?)\\s*:\\s*(.+?)\\s*(?:#\\s*(.*)|)$',  # "#   - name: value  # comment"
    postfix_comment_format='  # {}', comment_format='  # {}', comment_pattern='(?:\\s*#\\s*(.*)|)$'    # "#   # comment"
)
_header_comment_prefix_pattern = '^\\s*#'

def _select_while_matching(strings: Iterable[str], config_pattern: str, skip_blank_lines: bool = True) -> Iterable[str]:
    config_regex = re.compile(config_pattern)
    for string in strings:
        if skip_blank_lines and string.strip() == '':
            continue
        if not config_regex.match(string):
            return
        yield string


def _configuration_from_store(store, path='/configuration/', config: Optional[_cfg.Configuration] = None) -> _cfg.Configuration:
    if config is None:
        config = _cfg.Configuration()
    for lpath, groups, leaves in store.walk(path):
        for leave_name in leaves:
            df = store.get(lpath + '/' + leave_name)
            names = df['name'].to_numpy()
            values = df['value'].to_numpy()
            comments = df['comment'].to_numpy()
            section = _cfg.ConfigurationSection(name=leave_name)  # cannot serialize section comments
            for name, value, comment in zip(names, values, comments):
                section.add_entry(_cfg.ConfigurationEntry(name=name, value=value, comment=comment))
            config.override_section(section)
    return config

def _configuration_to_store(config: _cfg.Configuration, store, path='/configuration/') -> None:
    import pandas as pd
    for section in config:
        df = pd.DataFrame.from_dict({
            'name': [entry.name for entry in section],
            'value': [entry.value for entry in section],
            'comment': [entry.comment for entry in section]
        })
        store.put(path + section.name, df)


def _detect_format(filename: str) -> str:
    if filename.endswith('.tsv.gz'):
        return 'tsv+gzip'
    elif filename.endswith('.tsv'):
        return 'tsv'
    elif filename.endswith('.parquet'):
        return 'parquet'
    if filename.endswith('.hdf5.gz') or filename.endswith('.h5.gz') or filename.endswith('.hdf.gz'):
        return 'hdf5+gzip'
    if filename.endswith('.hdf5') or filename.endswith('.h5') or filename.endswith('.hdf'):
        return 'hdf5'
    else:
        raise RuntimeError("don't know which file format to assume for file '{}': no known file extension".format(filename))


def _load_data_file(filename: str, format: str, load_data: bool, load_config: bool) -> Tuple[_cfg.Configuration, Optional[pd.DataFrame]]:
    def load_config_from_tsv(text_file):
        if load_config:
            config_lines = _select_while_matching(strings=text_file, config_pattern=_header_comment_prefix_pattern, skip_blank_lines=True)
            config = _cfg.Configuration()
            config.from_strings(strings=config_lines, options=_header_comment_serialization_options)
            return config
        else:
            return None
    def load_data_from_tsv(text_file):
        return pd.read_csv(text_file, sep='\t', comment='#') if load_data else None
    def load_data_from_parquet(binary_file):
        return pd.read_parquet(binary_file, engine='pyarrow') if load_data else None
    def load_config_and_data_from_hdf5(filename):
        import h5py  # needed on Windows, otherwise DLL path issues ensue
        with pd.HDFStore(filename, mode='r') as store:
            config = _configuration_from_store(store, path='/configuration/') if load_config else None
            data = store['data'] if load_data else None
            return config, data

    if format == 'auto':
        format = _detect_format(filename=filename)

    try:
        if format == 'tsv+gzip':
            with gzip.open(filename, mode='rt') as text_file:
                config = load_config_from_tsv(text_file)
            with gzip.open(filename, mode='rt') as text_file:
                data = load_data_from_tsv(text_file)
            return config, data
        elif format == 'tsv':
            with open(filename, mode='r') as text_file:
                config = load_config_from_tsv(text_file)
            with open(filename, mode='r') as text_file:
                data = load_data_from_tsv(text_file)
            return config, data
        elif format == 'parquet':
            if load_config:
                raise RuntimeError('cannot load configuration data from Parquet file')
            with open(filename, mode='r') as binary_file:
                data = load_data_from_parquet(binary_file)
            return None, data
        elif format == 'hdf5+gzip':
            with DelayRemovedNamedTemporaryFile() as temp_file:
                with gzip.open(filename, mode='rb') as compressed_file:
                    shutil.copyfileobj(compressed_file, temp_file.file)
                temp_file.file.close()
                return load_config_and_data_from_hdf5(temp_file.name)
        elif format == 'hdf5':
            return load_config_and_data_from_hdf5(filename)
        else:
            raise RuntimeError("unknown data file format '{}'".format(format))
    except Exception as e:
        raise RuntimeError("error while reading data from file '{}': {}".format(filename, e))

def _save_data_file(config: _cfg.Configuration, data: Optional[pd.DataFrame], filename: str, format: str, force_dir: bool) -> None:
    def compress_and_remove(uncompressed_filename, filename):
        with open(uncompressed_filename, mode='rb') as uncompressed_file:
                with gzip.open(filename, mode='wb') as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
        os.unlink(uncompressed_filename)
    def save_config_and_data_to_tsv(text_file):
        header_lines = config.to_strings(options=_header_comment_serialization_options)
        for line in header_lines:
            text_file.write(line + '\n')
        data.to_csv(text_file, sep='\t', index=False)
    def save_data_to_parquet(filename, compression):
        data.to_parquet(filename, engine='pyarrow', compression=compression, index=False)
    def save_config_and_data_to_hdf5(hdf5_filename):
        import h5py  # needed on Windows, otherwise DLL path issues ensue
        with pd.HDFStore(hdf5_filename, mode='w', complevel=6) as store:
            _configuration_to_store(config, store, path='/configuration/')
            if data is not None:
                store.put('data', data)

    if format == 'auto':
        format = _detect_format(filename=filename)

    try:
        dir, _ = os.path.split(filename)
        if dir != '' and force_dir:
            pathlib.Path(dir).mkdir(exist_ok=True)
        if format == 'tsv+gzip':
            with gzip.open(filename, mode='wt') as compressed_file:
                save_config_and_data_to_tsv(compressed_file)
        elif format == 'tsv':
            save_config_and_data_to_tsv(filename)
        elif format == 'parquet+gzip':
            save_data_to_parquet(filename, compression='gzip')
        elif format == 'parquet':
            save_data_to_parquet(filename, compression='none')
        elif format == 'hdf5+gzip':
            hdf5_filename = filename + '.uncompressed'
            save_config_and_data_to_hdf5(hdf5_filename)
            compress_and_remove(hdf5_filename, filename)
        elif format == 'hdf5':
            save_config_and_data_to_hdf5(filename)
        else:
            raise RuntimeError("unknown data file format '{}'".format(format))
    except Exception as e:
        raise RuntimeError("error while saving configuration to file '{}': {}".format(filename, e))


def load_data_file_configuration(filename: str, format: str = 'auto') -> _cfg.Configuration:
    config, _ = _load_data_file(filename=filename, format=format, load_data=False, load_config=True)
    return config

def load_data_file_data(filename: str, format: str = 'auto') -> pd.DataFrame:
    _, data = _load_data_file(filename=filename, format=format, load_data=True, load_config=False)
    return data

def load_data_file(filename: str, format: str = 'auto') -> Tuple[_cfg.Configuration, pd.DataFrame]:
    return _load_data_file(filename=filename, format=format, load_data=True, load_config=True)

def save_data_file(config: _cfg.Configuration, data: Optional[pd.DataFrame], filename: str, format: str = 'auto') -> None:
    _save_data_file(config=config, data=data, filename=filename, format=format, force_dir=True)


def _unpack_filename(filename: str) -> Tuple[str, str]:  # name, filetype
    if filename == 'config.par':
        return 'config', 'configuration'
    elif filename.endswith('.numpy.txt'):
        return filename.removesuffix('.numpy.txt'), 'numpy-text'
    elif filename.endswith('.npy'):
        return filename.removesuffix('.npy'), 'npy'
    elif filename.endswith('.npz'):
        return filename.removesuffix('.npz'), 'npz'
    elif filename.endswith('.tsv.gz'):
        return filename.removesuffix('.tsv.gz'), 'tsv+gzip'
    elif filename.endswith('.tsv'):
        return filename.removesuffix('.tsv'), 'tsv'
    elif filename.endswith('.parquet'):
        return filename.removesuffix('.parquet'), 'parquet'
    else:
        return filename, 'unknown', 'unknown'

class TarArchive:
    class Entry:
        def __init__(self, tarfile, tarinfo, name, filetype):
            self._tarfile = tarfile
            self._tarinfo = tarinfo
            self.name = name
            self.filetype = filetype

        #def load_file(self):
        #    return self._tarfile.extractfile(self._tarinfo)

        def load_data(self):
            #with self.load_file() as file:
            #    return _load_data(self._tarinfo.name, file, self.filetype)
            raise RuntimeError("not implemented yet")

    def __init__(self, filename: str):
        self._tarfile = tarfile.open(filename, "r")
        try:
            members_filenames_filetypes = [
                (member, *_unpack_filename(member.name))
                for member in filter(lambda member: member.isfile(), self._tarfile.members())
            ]
            self.entries = {
                filename: TarArchive.Entry(self._tarfile, member, filename, filetype)
                for member, filename, filetype in members_filenames_filetypes
            }
            self.config = self.entries['config'].load_data()
        except:
            self._tarfile.close()
            raise

    def close(self):
        if self._tarfile is not None:
            self._tarfile.close()
            self._tarfile = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

class DirectoryArchive:
    class Entry:
        def __init__(self, filename, name, filetype):
            self._filename = filename
            self.name = name
            self.filetype = filetype

        def load_data(self):
            if self.filetype == 'configuration':
                return _cfg.load_configuration(filename=self._filename)
            elif self.filetype == 'tsv' or self.filetype == 'tsv+gzip':
                return pd.read_csv(self._filename, sep='\t', comment='#', compression='infer')
            elif self.filetype == 'parquet':
                return pd.read_parquet(path=self._filename, engine='pyarrow')
            elif self.filetype == 'numpy-text':
                return np.loadtxt(self._filename)
            elif self.filetype == 'npy' or self.filetype == 'npz':
                return np.load(self._filename)
            else:
                raise RuntimeError('error loading file "{}": unknown file type'.format(self._filename))

    @staticmethod
    def _raise(e):
        raise e

    def __init__(self, dirname: str):
        files = next(os.walk(dirname, onerror=DirectoryArchive._raise), (None, None, []))[2]
        fullpaths_filenames_filetypes = [
            (os.path.join(dirname, filename), *_unpack_filename(filename))
            for filename in files
        ]
        self.entries = {
            filename: DirectoryArchive.Entry(fullpath, filename, filetype)
            for fullpath, filename, filetype in fullpaths_filenames_filetypes
        }
        self.config = self.entries['config'].load_data()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

def load_archive(filename: str) -> Any:
    if os.path.isdir(filename):
        return DirectoryArchive(filename)
    elif os.path.isfile(filename):
        _, ext = os.path.splitext(filename)
        if ext == '.tar':
            return TarArchive(filename)
        else:
            # legacy data file support
            #cfg = load_data_file_configuration(filename)
            #...
            raise RuntimeError('loading legacy data files not currently implemented')
    else:
        raise RuntimeError('error loading archive from "{}": file or directory does not exist')

class DirectoryArchiveWriter:
    def __init__(self, dirname: str, config: _cfg.Configuration, overwrite: bool = False, prefer_binary: bool = False, compress: bool = True):
        self._dirname = dirname
        self._prefer_binary = prefer_binary
        self._compress = compress
        if not overwrite or not os.path.exists(dirname):
            os.mkdir(dirname)  # raises exception if directory does not exist
        _cfg.save_configuration(config, os.path.join(dirname, 'config.par'))

    def write_file(self, name: str, data: Any, filetype: str = 'auto'):
        compression_suffix = '+gzip' if self._compress else ''
        if filetype == 'auto':
            if isinstance(data, pd.DataFrame):
                filetype = 'parquet' if self._prefer_binary else 'tsv' + compression_suffix
            elif isinstance(data, Mapping):
                filetype = 'npz'
            elif isinstance(data, _cfg.Configuration):
                filetype = 'configuration'  # always a text file
            else:
                filetype = 'npy' if self._prefer_binary else 'numpy-text' + compression_suffix

        if filetype == 'configuration':
            filename = os.path.join(self._dirname, name + '.par')
            _cfg.save_configuration(data, filename)
        elif filetype == 'numpy-text':
            filename = os.path.join(self._dirname, name + '.numpy.txt')
            np.savetxt(filename)
        elif filetype == 'npy':
            filename = os.path.join(self._dirname, name + '.npy')
            np.save(filename, data)
        elif filetype == 'npz':
            filename = os.path.join(self._dirname, name + '.npz')
            if self._compress:
                np.savez_compressed(filename, **data)
            else:
                np.savez(filename, **data)
        elif filetype == 'tsv+gzip':
            filename = os.path.join(self._dirname, name + '.tsv.gz')
            data.to_csv(filename, sep='\t', index=False, compression='gzip')
        elif filetype == 'tsv':
            filename = os.path.join(self._dirname, name + '.tsv.gz')
            data.to_csv(filename, sep='\t', index=False)
        elif filetype == 'parquet':
            filename = os.path.join(self._dirname, name + '.parquet')
            compression = 'gzip' if self._compress else 'none'
            data.to_parquet(filename, engine='pyarrow', compression=compression, index=False)
        else:
            raise RuntimeError('unknown file type')

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
