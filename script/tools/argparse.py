
import argparse


def str2bool(v):
    # 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_boolean_argument(argparser, name_or_flags, default, *args, **kwargs):
    short = False
    if name_or_flags.startswith('--'):
        name = name_or_flags.removeprefix('--')
    elif name_or_flags.startswith('-'):
        name = name_or_flags.removeprefix('-')
        short = True
    else:
        name = name_or_flags
    if 'dest' in kwargs:
        dest = kwargs['dest']
    else:
        dest = name
    argparser.add_argument(name_or_flags, type=str2bool, nargs='?', const=True, default=default, *args, **kwargs)
    if not short:
        argparser.add_argument("--no-" + name, dest=dest, action='store_false')
