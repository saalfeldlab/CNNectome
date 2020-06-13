import pymongo
import lazy_property
import os
import csv


class CosemDB(object):
    def __init__(self, username, password, host, gt_version, training_version):
        self.host = host
        self.username = username
        self.password = password
        self.gt_version = gt_version
        self.training_version = training_version

    @lazy_property.LazyProperty
    def client(self):
        raise NotImplementedError

    def access(self, db_name, collection):
        raise NotImplementedError

    def get_all_validation_crops(self, output_type):
        raise NotImplementedError

    def get_crop_by_number(self, number):
        raise NotImplementedError

    def read_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                               metric_params):
        raise NotImplementedError

    def delete_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                 metric_params):
        raise NotImplementedError

    def write_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                metric_params, value):

        raise NotImplementedError

    def update_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                 metric_params, value):
        self.delete_evaluation_result(path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                      metric_params)
        self.write_evaluation_result(path,dataset, setup, iteration, labelname, cropno, threshold, metric,
                                     metric_params, value)


class MongoCosemDB(CosemDB):
    def __init__(self, username, password, host="cosem.int.janelia.org:27017", gt_version='v0003',
                 training_version='v0003.2'):
        super(MongoCosemDB, self).__init__(username, password, host, gt_version, training_version)

    @lazy_property.LazyProperty
    def client(self):
        client = pymongo.MongoClient(self.host, username=self.username, password=self.password)
        return client

    def access(self, db_name, collection):
        return self.client[db_name][collection]

    def get_crop_by_number(self, number):
        crop_db = self.access('crops', self.gt_version)
        crop = crop_db.find_one({"number": str(number)})
        return crop

    def get_all_validation_crops(self, output_type=list):
        crop_db = self.access('crops', self.gt_version)
        filter = {'completion': -1}
        skip = {'_id': 0}
        crops = crop_db.find(filter, skip)
        if output_type is not None:
            crops = output_type(crops)
        return crops

    def read_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                               metric_params):
        eval_db = self.access('evaluation', self.training_version)
        filter = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                  'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params}
        skip = {'_id': 0}
        eval = eval_db.find_one(filter, skip)
        return eval

    def delete_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                 metric_params):
        eval_db = self.access('evaluation', self.training_version)
        filter = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                  'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params}
        eval = eval_db.delete_one(filter)

    def write_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                metric_params, value):
        eval_db = self.access('evaluation', self.training_version)
        document = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                    'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params,
                    'value': value}
        id = eval_db.insert_one(document)
        return id

    def update_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                 metric_params, value):
        eval_db = self.access('evaluation', self.training_version)
        query_doc = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                    'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params}
        document = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                    'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params,
                    'value': value}
        eval_db.update_one(query_doc, document)


class CosemCSV(object):
    def __init__(self, csv_folder):
        self.folder = csv_folder
        self.fieldnames = ["path", "dataset", "setup", "iteration", "label", "crop", "threshold", "metric",
                           "metric_params", "value"]

    def read_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                               metric_params):
        query_element = {"path": path, "dataset": dataset, "setup": setup, "iteration": iteration, "label": labelname,
                         "crop": cropno, "threshold": threshold, "metric": metric, "metric_params": metric_params}
        with open(os.path.join(self.folder, labelname + '.csv', "r")) as f:
            reader = csv.DictReader(f, self.fieldnames)
            for row in reader:
                for k, v in query_element.items():
                    if row[k] != json.dumps(v):
                        break
                else:
                    query_element['value'] = json.loads(row["value"])
                    return query_element

    def delete_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric, metric_params):
        query_element = {"path": path, "dataset": dataset, "setup": setup, "iteration": iteration, "label": labelname,
                         "crop": cropno, "threshold": threshold, "metric": metric, "metric_params": metric_params}
        with open(os.path.join(self.folder, labelname+'.csv'), "r") as f:
            reader = csv.DictReader(f, self.fieldnames)
            all_rows = []
            for row in reader:
                for k, v in query_element.items():
                    if row[k] != v:
                        all_rows.append(row)
                        break
        self.erase(labelname)
        with open(os.path.join(self.folder, labelname+'.csv'), "w") as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def write_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric, metric_params,
                                value):
        open(os.path.join(self.folder, labelname + '.csv'), "a").close()
        with open(os.path.join(self.folder, labelname + '.csv'), "r+") as f:
            reader = csv.DictReader(f, self.fieldnames)
            try:
                next(reader)
            except StopIteration:
                writer = csv.DictWriter(f, self.fieldnames)
                writer.writeheader()

        with open(os.path.join(self.folder, labelname + '.csv'), "a+") as f:
            writer = csv.DictWriter(f, self.fieldnames)
            document = {'path': path, 'dataset': dataset, 'setup': setup, 'iteration': iteration, 'label': labelname,
                        'crop': cropno, 'threshold': threshold, 'metric': metric, 'metric_params': metric_params,
                        'value': value}
            writer.writerow(document)

    def update_evaluation_result(self, path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                 metric_params, value):
        self.delete_evaluation_result(path, dataset, setup, iteration, labelname, cropno, threshold, metric,
                                      metric_params)
        self.write_evaluation_result(path,dataset, setup, iteration, labelname, cropno, threshold, metric,
                                     metric_params, value)

    def erase(self, labelname):
        if os.path.exists(os.path.join(self.folder, labelname + '.csv')):
            os.remove(os.path.join(self.folder, labelname + '.csv'))




