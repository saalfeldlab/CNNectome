import argparse
from CNNectome.utils.hierarchy import hierarchy
from CNNectome.utils.cosem_db import MongoCosemDB, CosemCSV

db_host = "cosem.int.janelia.org:27017"
gt_version = "v0003"
training_version = "v0003.2"
eval_results_csv_folder = "/groups/cosem/cosem/computational_evaluation/{training_version:}/csv/".format(training_version=training_version)


def transfer(db_username, db_password):
    db = MongoCosemDB(db_username, db_password, host=db_host, gt_version=gt_version, training_version=training_version)
    eval_col = db.access("evaluation", training_version)
    csv_d = CosemCSV(eval_results_csv_folder)
    for l in hierarchy.keys():
        csv_d.erase(l)
    for db_entry in eval_col.find():
        csv_d.write_evaluation_result(db_entry)


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    args = parser.parse_args()
    transfer(args.db_username, args.db_password)


if __name__ == "__main__":
    main()
