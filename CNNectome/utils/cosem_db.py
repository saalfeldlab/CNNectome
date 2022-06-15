import pymongo
import lazy_property
import os
import csv
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Union
from CNNectome.utils import config_loader


class CosemDB(object):
    def __init__(self, uri, gt_version, training_version, write_access):
        self.uri = uri
        self.gt_version = gt_version
        self.training_version = training_version
        self.write_access = write_access

    @lazy_property.LazyProperty
    def client(self):
        raise NotImplementedError

    def access(self, db_name, collection):
        raise NotImplementedError

    def get_all_validation_crops(self, output_type):
        raise NotImplementedError

    def get_crop_by_number(self, number):
        raise NotImplementedError

    def get_validation_crop_by_cell_id(self, cell_id):
        raise NotImplementedError

    def find(self, query):
        raise NotImplementedError

    def read_evaluation_result(self, query):
        raise NotImplementedError

    def delete_evaluation_result(self, query):
        raise NotImplementedError

    def write_evaluation_result(self, document):
        raise NotImplementedError

    def update_evaluation_result(self, query, value):
        self.delete_evaluation_result(query)
        query["value"] = value
        self.write_evaluation_result(query)


class MongoCosemDB(CosemDB):
    def __init__(
        self,
        uri: Optional[str] = None,
        gt_version: str = "v0003",
        training_version: str = "v0003.2",
        write_access: bool = False,
    ):
        if uri is None:
            if write_access:
                uri = config_loader.get_config()["organelles"]["database-private"]
            else:
                uri = config_loader.get_config()["organelles"]["database-public"]

        super(MongoCosemDB, self).__init__(
            uri, gt_version, training_version, write_access
        )

    @lazy_property.LazyProperty
    def client(self):
        client = pymongo.MongoClient(self.uri)
        return client

    def access(self, db_name, collection=None):
        if collection is None:
            if db_name == "evaluation":
                collection = (self.training_version, self.gt_version)
            elif db_name == "crops":
                collection = self.gt_version
            else:
                raise ValueError(
                    f"No default value for collection for database {db_name:}"
                )
        if isinstance(collection, Iterable) and not isinstance(collection, str):
            collection = ".".join(collection)
        return self.client[db_name][collection]

    def get_crop_by_number(self, number: Union[int, str]):
        crop_db = self.access("crops", self.gt_version)
        crop = crop_db.find_one({"number": str(number)})
        return crop

    def get_validation_crop_by_cell_id(self, cell_id):
        crop_db = self.access("crops", self.gt_version)
        filter = {
            "completion": -1,
            "dataset_id": cell_id,
            "alias": {"$regex": "Validation"},
        }
        crop = crop_db.find_one(filter)
        return crop

    def get_all_validation_crops(self, output_type=list):
        crop_db = self.access("crops", self.gt_version)
        filter = {"completion": -1, "alias": {"$regex": "Validation"}}
        skip = crop_db.find_one()
        for k in skip.keys():
            skip[k] = True
        skip["_id"] = False
        crops = crop_db.aggregate(
            [{"$match": filter}, {"$sort": {"number": 1}}, {"$project": skip}]
        )
        if output_type is not None:
            crops = output_type(crops)
        return crops

    def find(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        eval_db = self.access("evaluation", (self.training_version, self.gt_version))
        result = []
        for qu in eval_db.find(query):
            result.append(qu)
        return result

    def read_evaluation_result(self, query):
        eval_db = self.access("evaluation", (self.training_version, self.gt_version))
        num = eval_db.count_documents(query)
        if num > 1:
            raise ValueError(
                "Query for reading evaluation results matched more than one database entry, consider "
                "using find instead. Used query was {0:}".format(query)
            )
        skip = {"_id": 0}
        eval = eval_db.find_one(query, skip)
        return eval

    def delete_evaluation_result(self, query):
        eval_db = self.access("evaluation", (self.training_version, self.gt_version))
        eval = eval_db.delete_many(query)

    def write_evaluation_result(self, document):
        eval_db = self.access("evaluation", (self.training_version, self.gt_version))
        id = eval_db.insert_one(document.copy())
        return id

    def update_evaluation_result(self, query, value):
        eval_db = self.access("evaluation", (self.training_version, self.gt_version))
        document = query.copy()
        document["value"] = value
        num = eval_db.count_documents(query)
        if num > 1:
            raise ValueError(
                "Query for updating evaluation results matched more than one database entry. Used query "
                "was {0:}".format(query)
            )
        eval_db.update_one(query, document)


class CosemCSV(object):
    def __init__(self, csv_folder):
        self.folder = csv_folder
        self.fieldnames = [
            "path",
            "dataset",
            "setup",
            "iteration",
            "label",
            "crop",
            "raw_dataset",
            "parent_path",
            "parent_dataset_id",
            "threshold",
            "refined",
            "metric",
            "metric_params",
            "value",
        ]

    def read_evaluation_result(self, query):
        with open(os.path.join(self.folder, query["label"] + ".csv", "r")) as f:
            reader = csv.DictReader(f, self.fieldnames)
            for row in reader:
                for k, v in query.items():
                    if k in self.fieldnames:
                        if row[k] != json.dumps(v):
                            break
                    else:
                        logging.debug(
                            "Ignoring key {0:} for querying from csv".format(k)
                        )
                else:
                    query["value"] = json.loads(row["value"])
                    return query

    def delete_evaluation_result(self, query):
        with open(os.path.join(self.folder, query["label"] + ".csv"), "r") as f:
            reader = csv.DictReader(f, self.fieldnames)
            all_rows = []
            for row in reader:
                for k, v in query.items():
                    if k in self.fieldnames:
                        if row[k] != v:
                            all_rows.append(row)
                            break
                    else:
                        logging.debug(
                            "Ignoring key {0:} for querying from csv".format(k)
                        )
        self.erase(query["label"])
        with open(os.path.join(self.folder, query["label"] + ".csv"), "w") as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def write_evaluation_result(self, document):
        docid = document.pop("_id", None)
        open(os.path.join(self.folder, document["label"] + ".csv"), "a").close()
        with open(os.path.join(self.folder, document["label"] + ".csv"), "r+") as f:
            reader = csv.DictReader(f, self.fieldnames)
            try:
                next(reader)
            except StopIteration:
                writer = csv.DictWriter(f, self.fieldnames)
                writer.writeheader()

        with open(os.path.join(self.folder, document["label"] + ".csv"), "a+") as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writerow(document)

        if docid is not None:
            document["_id"] = docid

    def erase(self, labelname):
        if os.path.exists(os.path.join(self.folder, labelname + ".csv")):
            os.remove(os.path.join(self.folder, labelname + ".csv"))
