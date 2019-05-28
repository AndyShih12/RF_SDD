"""JSON minify program. """

import json # import json library
import sys # import sys library

def minify(config_file, config_minify_file):
  "Minify JSON"
  input_lines = open(config_file, "r", 1).readlines() # store file info in variable

  output_lines = ""
  for line in input_lines:
    ind = line.find("//")
    if ind > -1:
      line = line[:ind]
    output_lines += line + "\n"

  json_data = json.loads(output_lines) # store in json structure
  json_string = json.dumps(json_data, indent=2, sort_keys=True)
  json_string += "\n"

  open(config_minify_file, "w+", 1).write(json_string) # open and write json_string to file


config_file = sys.argv[1] # get arguments passed to command line excluding first arg
config_minify_file = sys.argv[2]

minify(config_file, config_minify_file)
