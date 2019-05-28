def read_constraints(constraint_filename):
    with open(constraint_filename, "r") as f:
        lines = f.readlines()
    lines = lines[1:]

    constraints = [tuple(x.strip().split(" ")) for x in lines]
    return constraints

def convert_float_data_to_binary_data(float_data, constraints):
  binary_header = []
  binary_data = []
  
  for i,c in enumerate(constraints):
    f = float_data[i] if i < len(float_data) else None
    l = int(c[1])
    offset = 2

    for j in xrange(l):
      if f == None:
        binary_data.append(None)
      elif f >= float(c[j+offset]):
        binary_data.append(1)
      else:
        binary_data.append(0)
      binary_header.append(c[0] % j)

  return binary_header, binary_data

def convert_binary_data_to_float_data(binary_data, constraints, zero=0, one=1):
  float_header = []
  float_data = []
  index = 0
  for c in constraints:
    zero_index, one_index = -1, -1
    for j in xrange(int(c[1])):
      cur = index + j
      if binary_data[cur] == zero and zero_index == -1:
        zero_index = j
      if binary_data[cur] == one:
        one_index = j
    index = index + int(c[1])

    offset = 2
    lb, ub = None, None
    if zero_index != -1:
      ub = float(c[zero_index + offset])
    if one_index != -1:
      lb = float(c[one_index + offset])
    float_header.append(c[0][:-3])
    float_data.append((lb, ub))
  return float_header, float_data
