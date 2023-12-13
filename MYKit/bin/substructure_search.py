from argparse import ArgumentParser
import glob
from pathlib import Path
import sys

p = str(Path(__file__).parent.parent.resolve())

print(p)

sys.path.insert(0, p)
import MYKit

if __name__ == "__main__":
    parser = ArgumentParser(
        "Given an sdf file, search other sdf files for matching substructure"
    )

    parser.add_argument(
        "-q",
        "--query_structure",
        type=str,
        required=True,
        help="SDF file for structure to query for in other molecules",
    )
    parser.add_argument(
        "-d",
        "--database_structures",
        type=str,
        required=True,
        help="A file glob matching SDF files to search",
    )
    parser.add_argument(
        "--min_path", type=int, default=2, help="minimum path length for fingerprint"
    )
    parser.add_argument(
        "--max_path", type=int, default=7, help="minimum path length for fingerprint"
    )
    parser.add_argument(
        "-H",
        "--include_hydrogen",
        type=bool,
        default=False,
        help="Include hydrogen in structures",
    )

    args = parser.parse_args()

    ##############################################

    # Read in query
    query = MYKit.SDFToMol(args.query_structure, include_hydrogen=args.include_hydrogen)

    # Get molecules to search
    for other_sdf in glob.glob(args.database_structures):
        other = MYKit.SDFToMol(other_sdf, include_hydrogen=args.include_hydrogen)
        try:
            if other.hasSubstructMatch(query):
                print("Found match: ", other_sdf)
        except:
            print("ERROR ON FILE: ", other_sdf)
