
class Entry:
  def __init__(self, lineno, data, leader, category):
    self.line = lineno
    self.leader = leader
    self.data = data
    self.category = category
    for d in data:
      assert type(d) is int or type(d) is float, (d, data, lineno, category.filename)


class Category:
  def __init__(self, name, filename):
    self.name = name
    self.entries = []
    self.filename = filename


  def add(self, lineno, data, leader=None):
    e = Entry(lineno, data, leader, self)
    self.entries.append(e)


  def size(self):
    return len(self.entries)


  def compare(self, other):
    m = min(self.size(), other.size())
    print "Category", self.name
    print ("=" * len("Category " + self.name))
    print self.filename, ":", self.size()
    print other.filename, ":", other.size()
    if m < self.size():
      print filename, "has", (self.size() - m),  "more entries"
    elif m < other.size():
      print other.filename, "has", (other.size() - m),  "more entries"

    bad = 0
    different_length = 0
    eps = 1e-9
    for index, (x, y) in enumerate(zip(self.entries[0:m], other.entries[0:m])):
      if len(x.data) != len(y.data):
        if different_length < 5:
          print "  Entry", index, ":data length different", len(x.data), \
              "(line", x.line, ") vs", len(y.data), "(line", y.line, ")"
        different_length += 1
      else:
        z = [x.data[i] - y.data[i] for i in xrange(len(x.data))]
        z = [i if i > 0 else -i for i in z]
        pos = [i for i in xrange(len(z)) if z[i] > eps]
        if len(pos) > 0:
          if bad < 5:
            i = pos[0]
            print "  Entry", index, ":", len(pos), " indices differ by >", \
                eps, "e.g. index", i, \
                ", entry1 on line", x.line, "has", x.data[i], \
                ", entry2 on line", y.line, "has", y.data[i]
          bad += 1
    if different_length == 0:
      print "Overall", bad, "differing entries\n"
    else:
      print different_length, "entries with differing lengths\n"


class Reader:
  def __init__(self, filename):
    self.categories = []
    with open(filename, "r") as f:
      count = 0
      ignore = False
      for line in f:
        line = line.strip()
        if line.find("StartAnnotation") != -1:
          ignore = True
        if line.find("EndAnnotation") != -1:
          ignore = False
        count += 1
        if ignore: continue

        if line.startswith("INFO:tensorflow:Debug"):
          line = line[16:]
        if line.startswith("Debug="):
          parts = line.split("=")
          parts = [p.strip() for p in parts]
          assert len(parts) >= 3, line

          catname = parts[1]
          catname = catname.replace("-", "_")
          if catname == "Oracle": continue

          leader = None
          if len(parts) == 4:
            leader = parts[2]

          data = parts[-1]
          data = data.replace("[[", "[").replace("]]", "]")
          if data.find(",") == -1:
            data = data.replace(" ", ",")

          ls = None
          if data.startswith("("):
            ls = eval(data)
            assert type(ls) is tuple, data
            assert len(ls) == 1, data
            ls = [ls[0]]
          elif not data.startswith("["):
            ls = eval(data)
            assert type(ls) is int or type(ls) is float, data
            ls = [ls]
          else:
            ls = eval(data)

          assert type(ls) is list, data
          if len(ls) > 0:
            t = type(ls[0])
            assert t is float or t is int, (str(t), ls[0], line)

          if parts[1].find("Fixed_Ids") != -1:
            ls = [d for d in ls if d != -1]

          category = self.get(catname)
          if category is None:
            category = Category(catname, filename)
            self.categories.append(category)
          category.add(count, ls, leader)

    print filename
    for c in self.categories:
      print " ", c.name, ":", c.size()


  def get(self, name):
    for c in self.categories:
      if c.name == name:
        return c
    return None

  def has(self, name):
    return self.get(name) is not None


class Debugger:
  def __init__(self):
    self.tf = None
    self.py = None


  def load(self, tf_file, py_file):
    self.tf = Reader(tf_file)
    self.py = Reader(py_file)


  def compare(self):
    for cat in self.tf.categories:
      other = self.py.get(cat.name)
      if other is not None:
        print
        cat.compare(other)
      else:
        print "Category", cat.name, "only in TF (", cat.size(), ")"

    for cat in self.py.categories:
      if not self.tf.has(cat.name):
        print "Category", cat.name, "only in PyTorch (", cat.size(), ")"


debugger = Debugger()
debugger.load("/path/to/tensorflow/log", "/path/to/pytorch/log")
debugger.compare()
