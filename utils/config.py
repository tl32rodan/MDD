from easydict import EasyDict as edict
import yaml

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    #for x in parser:
    #    print('{}: {}'.format(x, parser[x]))
    return parser
