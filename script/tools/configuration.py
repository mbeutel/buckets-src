

# Defines the classes `ConfigurationEntry`, `ConfigurationSection`, and `Configuration` to represent basic configuration data.
# Implements serialization and deserialization to/from strings.


import os
import re
import sys
import pathlib
from typing import Optional, Iterable


class ConfigSerializationOptions:
    """Controls how a configuration is serialized."""
    def __init__(self,
            short: bool,
            section_format: str, section_pattern: str, section_separator: bool,
            name_value_format: str, name_value_pattern: str,
            prefix: str = '', prefix_pattern: str = None,
            postfix_comment_format: str = '', comment_format: str = '', comment_pattern: Optional[str] = None):
        prefix_pattern_or_empty = prefix_pattern if prefix_pattern is not None else ''
        self.short = short
        self.section_format = prefix + section_format
        self.section_regex = re.compile(prefix_pattern_or_empty + section_pattern)
        self.section_separator = section_separator
        self.name_value_format = prefix + name_value_format
        self.name_value_regex = re.compile(prefix_pattern_or_empty + name_value_pattern)
        self.prefix = prefix.rstrip()
        self.prefix_regex = re.compile(prefix_pattern_or_empty + prefix_pattern) if prefix_pattern is not None else None
        self.postfix_comment_format = postfix_comment_format
        self.comment_format = prefix + comment_format
        self.comment_regex = re.compile(prefix_pattern_or_empty + comment_pattern) if comment_pattern is not None else None


def _select_representation(value: str, short_value: str, short: bool) -> str:
    if short:
        return short_value if short_value != '' else value
    else:
        return value if value != '' else short_value

def _append_comment(string: str, comment: str, options: ConfigSerializationOptions) -> str:
    if string != '':
        if comment != '':
            return string + options.postfix_comment_format.format(comment)
        else:
            return string
    else:
        if comment != '':
            return options.postfix_comment_format.format(comment)
        else:
            return options.prefix


class ConfigurationEntry:
    def __init__(self, name: str = '', short_name: str = '', value: str = '', short_value: str = '', comment: str = ''):
        assert not (name == '' and short_name != '')
        assert not (value == '' and short_value != '')

        self.name = name
        self.short_name = short_name
        self.value = value
        self.short_value = short_value
        self.comment = comment

    @staticmethod
    def from_string(string: str, options: ConfigSerializationOptions):
        key_value_match = options.name_value_regex.match(string)
        if key_value_match is not None:
            groups = key_value_match.groups()
            return ConfigurationEntry(name=groups[0], value=groups[1], comment=groups[2] if groups[2] is not None else '')
        else:
            comment_match = options.comment_regex.match(string) if options.comment_regex is not None else None
            if comment_match is not None:
                groups = comment_match.groups()
                if groups[0] is None:
                    return None  # blank line
                else:
                    return ConfigurationEntry(comment=groups[0])
            else:
                raise RuntimeError('cannot parse configuration line \'{}\': unrecognized syntax'.format(string))

    def to_string(self, options: ConfigSerializationOptions) -> str:
        any_name = _select_representation(self.name, self.short_name, options.short)
        any_value = _select_representation(self.value, self.short_value, options.short)
        if any_name == '':
            assert any_value == ''
            if self.comment == '':
                return options.prefix
            else:
                return options.comment_format.format(self.comment)
        else:
            line = options.name_value_format.format(any_name, any_value)
            return _append_comment(string=line, comment=self.comment, options=options)

class ConfigurationSection:
    def __init__(self, name: str = '', short_name: str = '', comment: str = '', entries: Iterable[ConfigurationEntry] = []):
        assert not (name == '' and short_name != '')
        
        self.name = name
        self.short_name = short_name
        self.comment = comment
        self._entries = []
        self._entry_dict = {}
        for entry in entries:
            self.add_entry(entry)

    def add_entry(self, entry: ConfigurationEntry) -> None:
        if entry.name in self._entry_dict or entry.short_name in self._entry_dict:
            raise RuntimeError("entry for '{}' already exists in the configuration section".format(entry.name))
        if entry.name != '':
            self._entry_dict[entry.name] = entry
        if entry.short_name != '':
            self._entry_dict[entry.short_name] = entry
        self._entries.append(entry)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entry_dict

    def __getitem__(self, name: str) -> ConfigurationEntry:
        return self._entry_dict[name]

    def override_entry(self, entry: ConfigurationEntry) -> None:
        as_name = entry.name in self._entry_dict
        as_short_name = entry.short_name in self._entry_dict
        if as_name:
            old_entry = self._entry_dict[entry.name]
        elif as_short_name:
            old_entry = self._entry_dict[entry.short_name]
        else:
            old_entry = None
        if old_entry is not None:
            old_entry.value = entry.value
            old_entry.short_value = entry.short_value
            if entry.comment != '':
                old_entry.comment = entry.comment
            if not as_name and entry.name != '':
                self._entry_dict[entry.name] = old_entry
            if not as_short_name and entry.short_name != '':
                self._entry_dict[entry.short_name] = old_entry
        else:
            self.add_entry(entry)

    def purge_comments(self) -> None:
        self.comment = ''
        self._entries = list(filter(lambda entry: entry.name != '', self._entries))
        for entry in self._entries:
            entry.comment = ''


class Configuration:
    def __init__(self, sections: Iterable[ConfigurationSection] = {}):
        self._sections = []
        self._section_dict = { }
        for section in sections:
            self.add_section(section)

    def add_section(self, section: ConfigurationSection) -> None:
        if section.name in self._section_dict:
            if section.name != '':
                raise RuntimeError("section '{}' already exists in the configuration".format(section.name))
            else:
                raise RuntimeError("unnamed section already exists in the configuration")
        if section.name == '':
            self._section_dict[''] = section
        else:
            self._section_dict[section.name] = section
            if section.short_name != '':
                self._section_dict[section.short_name] = section
        if section.name == '':
            self._sections.insert(0, section)  # an unnamed section must be the first section in the configuration
        else:
            self._sections.append(section)

    def __iter__(self):
        return iter(self._sections)

    def __len__(self):
        return len(self._sections)

    def __contains__(self, name: str) -> bool:
        return name in self._section_dict

    def __getitem__(self, name: str) -> ConfigurationSection:
        return self._section_dict[name]

    def override_section(self, section: ConfigurationSection):
        as_name = section.name in self._section_dict
        as_short_name = section.short_name in self._section_dict
        if as_name:
            old_section = self._section_dict[section.name]
        elif as_short_name:
            old_section = self._section_dict[section.short_name]
        else:
            old_section = None
        if old_section is not None:
            for entry in section:
                old_section.override_entry(entry)
            if section.comment != '':
                old_section.comment = section.comment
            if not as_name:
                self._section_dict[section.name] = old_section
            if not as_short_name and section.short_name != '':
                self._section_dict[section.short_name] = old_section
        else:
            self.add_section(section)

    def override(self, with_config):
        for section in with_config:
            self.override_section(section)

    def purge_comments(self) -> None:
        for section in self._sections:
            section.purge_comments()
     
    def from_strings(self, strings: Iterable[str], options: ConfigSerializationOptions) -> None:
        section = ConfigurationSection()
        for string in strings:
            if string.strip() != '':
                section_match = options.section_regex.match(string)
                if section_match is None:
                    entry = ConfigurationEntry.from_string(string, options=options)
                    if entry is not None:
                        section.add_entry(entry)
                else:
                    if section.name != '' or len(section) > 0:
                        self.override_section(section)
                    groups = section_match.groups()
                    section = ConfigurationSection(name=groups[0], comment=groups[1] if groups[1] is not None else '')
        if section.name != '' or len(section) > 0:
            self.override_section(section)

    def to_strings(self, options: ConfigSerializationOptions) -> Iterable[str]:
        first_section = True
        for section in self._sections:
            if not first_section and options.section_separator:
                yield options.prefix
            first_section = False
            any_name = _select_representation(section.name, section.short_name, options.short)
            if any_name != '' or section.comment != '':
                line = options.section_format.format(any_name) if any_name != '' else ''
                yield _append_comment(string=line, comment=section.comment, options=options)
            for entry in section:
                yield entry.to_string(options=options)


par_serialization_options = ConfigSerializationOptions(
    short=False,
    section_separator=True,
    section_format='{}:', section_pattern='^\\s*(.+?)\\s*:\\s*(?:#\\s*(.*)|)$',                             # "section:  # comment"
    name_value_format='  - {}: {}', name_value_pattern='^\\s*-\\s*(.+?)\\s*:\\s*(.+?)\\s*(?:#\\s*(.*)|)$',  # "  - name: value  # comment"
    postfix_comment_format='  # {}', comment_format='  # {}', comment_pattern='^(?:\\s*#\\s*(.*)|)$'        # "  # comment"
)

def _detect_format(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    lext = ext.lower()
    if lext == '.par':
        return 'par'
    else:
        raise RuntimeError("don't know which file format to assume for file '{}': no known file extension".format(filename))


short_serialization_options = ConfigSerializationOptions(
    short=True,
    section_separator=False,
    section_format='[{}]', section_pattern='^\\*\\[\\]\\s*$',                # [short_section]
    name_value_format='{}={}', name_value_pattern='^\\s*(.+?)[=](.+?)\\s*$'  # short_name=short_value
)

def _skip_blanks(strings: Iterable[str]) -> Iterable[str]:
    for string in strings:
        if string.strip() != '':
            yield string


def to_short_string(config: Configuration) -> str:
    return ' '.join(_skip_blanks(config.to_strings(options=short_serialization_options)))


def load_configuration(filename: str, format: str = 'auto', config: Optional[Configuration] = None) -> Configuration:
    if format == 'auto':
        format = _detect_format(filename=filename)

    try:
        if config is None:
            config = Configuration()
        if format == 'par':
            with open(filename, mode='r') as text_file:
                config.from_strings(strings=text_file, options=par_serialization_options)
            return config
        else:
            raise RuntimeError("unknown data file format '{}'".format(format))
    except Exception as e:
        raise RuntimeError("error while reading configuration from file '{}': {}".format(filename, e))

def save_configuration(config: Configuration, filename: str, format: str = 'auto') -> None:
    if format == 'auto':
        format = _detect_format(filename=filename)

    try:
        dir, _ = os.path.split(filename)
        if dir != '':
            pathlib.Path(dir).mkdir(exist_ok=True)
        if format == 'par':
            header_lines = config.to_strings(options=par_serialization_options)
            with open(filename, mode='w') as text_file:
                for line in header_lines:
                    print(line, file=text_file)
        else:
            raise RuntimeError("unknown data file format '{}'".format(format))
    except Exception as e:
        raise RuntimeError("error while saving configuration to file '{}': {}".format(filename, e))


def report(config: Configuration, file=sys.stdout) -> None:
    for line in config.to_strings(par_serialization_options):
        print(line, file=file)
