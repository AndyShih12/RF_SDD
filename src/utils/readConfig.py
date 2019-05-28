import json

def read_json(filename):
  with open(filename) as f:
    config_json = json.load(f)

  # add str() to convert from unicode to ascii
  dataset = config_json["dataset"]
  config_json["train_filename"] = str(config_json["train_filename"] % (dataset, dataset))
  config_json["test_filename"] = str(config_json["test_filename"] % (dataset, dataset))
  config_json["rf_basename"] = str(config_json["rf_basename"] % (dataset, dataset))
  config_json["binarized_rf_basename"] = str(config_json["binarized_rf_basename"] % (dataset, dataset))

  config_json["binarized_train_filename"] = str(config_json["binarized_train_filename"] % (dataset, dataset))
  config_json["binarized_test_filename"] = str(config_json["binarized_test_filename"] % (dataset, dataset))
  config_json["discretized_train_filename"] = str(config_json["discretized_train_filename"] % (dataset, dataset))
  config_json["discretized_test_filename"] = str(config_json["discretized_test_filename"] % (dataset, dataset))

  config_json["constraint_filename_working"] = str(config_json["constraint_filename_working"] % (dataset, dataset))
  config_json["constraint_filename_output"] = str(config_json["constraint_filename_output"] % (dataset, dataset))

  config_json["sdd_filename"] = str(config_json["sdd_filename"] % (dataset, dataset))
  config_json["vtree_filename"] = str(config_json["vtree_filename"] % (dataset, dataset))
  config_json["constraint_sdd_filename"] = str(config_json["constraint_sdd_filename"] % (dataset, dataset))
  config_json["constraint_vtree_filename"] = str(config_json["constraint_vtree_filename"] % (dataset, dataset))

  json_string = json.dumps(config_json, indent=2, sort_keys=True)
  json_string += "\n"

  return config_json
