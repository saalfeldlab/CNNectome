import zarr
import argparse
import os
import json


def add_attributes(path, version):
    if not os.path.exists(path):
        raise ValueError("Path {0:} does not exist".format(path))
    version_attr = {"n5": version}
    for dir, subdirlist, filelist in os.walk(path):
        if 'attributes.json' in filelist:
            del subdirlist[:]
        else:
            with open(os.path.join(dir, 'attributes.json'), 'w') as f:
                json.dump(version_attr, f)

    if not os.path.exists(os.path.join(path, 'attributes.json')):
        with open(os.path.join(path, 'attributes.json'), "w") as f:
            json.dump(version_attr, f)


def main():
    parser = argparse.ArgumentParser("Hacky script to add attributes.json to n5 root and every group to make it "
                                     "readable by zarr. Traverses directory tree and adds attributes.json in each "
                                     "subdirectory (group) until an attriubtes.json already exists")
    parser.add_argument("path", type=str, help="Path to the n5 container (root)")
    parser.add_argument("--version", type=str, help="n5-version to specify in each added attributes.json", default="2.0.0")
    args = parser.parse_args()
    add_attributes(args.path, args.version)


if __name__ == "__main__":
    main()