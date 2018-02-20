# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random


# Class for computing and serving a lexicon.
# Usage:
#   lexicon = Lexicon(min_count=..., normalize_digits=...)
#   lexicon.add("foo")
#   lexicon.add("bar")
#   lexicon.finalize()
#   ...
#   index = lexicon.index("foo")
class Lexicon:
  def __init__(self, min_count=1, normalize_digits=True, oov_item="<UNKNOWN>"):
    self.min_count = min_count
    self.counts = {}
    self.oov_item = oov_item
    self.oov_index = 0   # OOV is always at position 0
    self.normalize_digits = normalize_digits
    self.item_to_index = {}
    self.index_to_item = {}


  def _key(self, item):
    if self.normalize_digits:
      return "".join([c if not c.isdigit() else '9' for c in list(item)])
    else:
      return item


  def add(self, item):
    item = self._key(item)
    if item not in self.counts:
      self.counts[item] = 0
    self.counts[item] = self.counts[item] + 1


  def finalize(self):
    self.index_to_item[self.oov_index] = self.oov_item
    self.item_to_index[self.oov_item] = self.oov_index
    for item, count in self.counts.iteritems():
      if count >= self.min_count and item not in self.item_to_index:
        index = len(self.index_to_item)
        self.item_to_index[item] = index
        self.index_to_item[index] = item
    print len(self.item_to_index), "items in lexicon, including OOV"


  def size(self):
    return len(self.item_to_index)


  def index(self, item):
    item = self._key(item)
    if item not in self.item_to_index:
      return self.oov_index
    else:
      return self.item_to_index[item]


  def value(self, index):
    assert index >= 0 and index < len(self.index_to_item), "%r" % index
    return self.index_to_item[index]


# Mention comparator function.
def mention_comparator(x, y):
  b1 = x.begin
  b2 = y.begin
  if b1 < b2 or b1 == b2 and x.length > y.length:
    return -1
  if b1 > b2 or x.length < y.length:
    return 1
  return 0


# Stores a document and its gold transitions.
class Document:
  def __init__(self, commons, schema, encoded):
    self.store = sling.Store(commons)
    self.object = self.store.parse(encoded, binary=True)
    self.inner = sling.Document(frame=self.object, schema=schema)
    self.gold = []  # sequence of gold transitions

    self.tokens = self.inner.tokens
    self.mentions = self.inner.mentions
    self.themes = self.inner.themes
    self.inner.mentions.sort(cmp=mention_comparator)


  def size(self):
    return len(self.tokens)


# A set of documents.
class Corpora:
  def __init__(self):
    self.documents = []


  def read(self, recordio_filename, commons, schema, max_count=None):
    reader = sling.RecordReader(recordio_filename)
    for _, value in reader:
      self.add(Document(commons, schema, value))
      if max_count is not None and max_count <= len(self.documents): break
    reader.close()


  def size(self):
    return len(self.documents)


  def add(self, doc):
    self.documents.append(doc)


  def shuffle(self):
    random.shuffle(self.documents)


  def subset(self, start, end):
    s = Corpora()
    s.documents = self.documents[start:end]
    return s


# Represents a single transition.
class Action:
   SHIFT = 0
   STOP = 1
   EVOKE = 2
   REFER = 3
   CONNECT = 4
   ASSIGN = 5
   EMBED = 6
   ELABORATE = 7
   NUM_ACTIONS = 8

   def __init__(self, t=None):
     self.type = None
     self.length = None
     self.source = None
     self.target = None
     self.role = None
     self.label = None

     if t is not None:
       assert t < Action.NUM_ACTIONS
       assert t >= 0
       self.type = t


   def _totuple(self):
     return (self.type, self.length, self.source, self.target,
             self.role, self.label)


   def __hash__(self):
     return hash(self._totuple())


   def __eq__(self, other):
     return self._totuple() == other._totuple()


   def __repr__(self):
     names = {
       Action.SHIFT: "SHIFT",
       Action.STOP: "STOP",
       Action.EVOKE: "EVOKE",
       Action.REFER: "REFER",
       Action.CONNECT: "CONNECT",
       Action.ASSIGN: "ASSIGN",
       Action.EMBED: "EMBED",
       Action.ELABORATE: "ELABORATE"
     }

     s = names[self.type]
     for k, v in sorted(self.__dict__.iteritems()):
       if v is not None and k != "type":
         if s != "":
           s = s + ", "
         s = s + k + ": " + str(v)
     return s


# Outputs a list of transitions that represent a given document's frame graph.
class TransitionGenerator:
  # Bookkeeping for one frame.
  class FrameInfo:
    def __init__(self, handle):
      self.handle = handle
      self.type = None
      self.edges = []
      self.mention = None
      self.output = False


  # Labeled edge between two frames. Each edge is used to issue a CONNECT,
  # EMBED, ELABORATE, or ASSIGN action.
  class Edge:
    def __init__(self, incoming=None, role=None, value=None):
      self.incoming = incoming
      self.role = role
      self.neighbor = value
      self.inverse = None
      self.used = False


  # Rough action that will be translated into an Action later on.
  class RoughAction:
    def __init__(self, t=None):
      self.action = Action(t)
      self.info = None
      self.other_info = None


  def __init__(self, commons):
    self.commons = commons
    self._id = commons["id"]
    self._isa = commons["isa"]
    self._is = commons["is"]
    self._thing_type = commons["/s/thing"]  # fallback type
    assert self._thing_type.isglobal()


  def is_ref(self, handle):
    return type(handle) == sling.Frame


  # Creates a FrameInfo object for 'frame' and recursively for all frames
  # pointed to by it.
  def _init_info(self, frame, frame_info, initialized):
    if frame in initialized:
      return
    initialized[frame] = True

    info = frame_info.get(frame, None)
    if info is None:
      info = FrameInfo(frame)
      frame_info[frame] = info

    pending = []
    for role, value in frame:
      if not self.is_ref(value) or role == self._id
        or (role == self._isa and value.islocal()):
        continue
      if role == self._isa and info.type is None:
        info.type = value
      else:
        edge = Edge(incoming=False, role=role, value=value)
        info.edges.append(edge)
        if value == frame:
          edge.inverse = edge
          continue

        if value.islocal():
          nb_info = frame_info.get(value, None)
          if nb_info is None:
            nb_info = FrameInfo(value)
            frame_info[value] = nb_info
          nb_edge = Edge(incoming=True, role=role, value=frame)
          nb_info.edges.append(nb_edge)
          nb_edge.inverse = edge
          edge.inverse = nb_edge
          pending.append(value)

    if info.type is None:
      info.type = self._thing_type

    for p in pending:
      self._init_info(p, frame_info, initialized)


  # Translates rough action 'rough' to an Action using indices from 'attention'.
  def _translate(self, attention, rough):
    action = Action(t=rough.action.type)
    if rough.action.length is not None:
      action.length = rough.action.length
    if rough.action.role is not None:
      action.role = rough.action.role

    if action.type == Action.EVOKE:
       action.label = rough.info.type
    elif action.type == Action.REFER:
      action.target = attention.index(rough.info.handle)
    elif action.type == Action.EMBED:
      action.label = rough.info.type
      action.target = attention.index(rough.other_info.handle)
    elif action.type == Action.ELABORATE:
      action.label = rough.info.type
      action.source = attention.index(rough.other_info.handle)
    elif action.type == Action.CONNECT:
      action.source = attention.index(rough.info.handle)
      action.target = attention.index(rough.other_info.handle)
    elif action.type == Action.ASSIGN:
      action.source = attention.index(rough.info.handle)
      action.label = rough.action.label

    return action


  def _update(self, attention, rough):
    t = rough.action.type
    if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
      attention.insert(0, rough.info.handle)
    elif t == Action.REFER or t == Action.ASSIGN or t == Action.CONNECT:
      attention.remove(rough.info.handle)
      attention.insert(0, rough.info.handle)


  def generate(self, document):
    frame_info = {}
    initialized = {}
    for m in document.mentions:
      for evoked in m.evokes():
        self.init_info(evoked, frame_info, initialized)
        frame_info[evoked].mention = m

    for theme in document.themes:
      self.init_info(theme, frame_info, initialized)

    rough_actions = []
    start = 0
    evoked = {}
    for m in document.mentions:
      for i in xrange(start, m.begin):
        rough_actions.append(RoughAction(Action.SHIFT))
      start = m.begin

      for frame in m.evokes():
        rough_action = RoughAction()
        rough_action.action.length = m.length
        rough_action.info = frame_info[frame]
        if frame not in evoked:
          rough_action.action.type = Action.EVOKE
          evoked[frame] = True
        else:
          rough_action.action.type = Action.REFER
        rough_actions.append(rough_action)

    for index in xrange(start, len(document.tokens)):
      rough_actions.append(RoughAction(Action.SHIFT))

    rough_actions.append(RoughAction(Action.STOP))
    rough_actions.reverse()
    actions = []
    attention = []
    while len(rough_actions) > 0:
      rough_action = rough_actions.pop()
      actions.append(self._translate(attention, rough_action))
      self._update(attention, rough_action)
      t = rough_action.action.type
      if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
        rough_action.info.output = True

        # CONNECT actions for the newly output frame.
        for e in rough_action.info.edges:
          if e.used: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and nb.output:
            connect = RoughAction(Action.CONNECT)
            connect.action.role = e.role
            connect.info = nb if e.incoming else rough_action.info
            connect.other_info = rough_action.info if e.incoming else nb
            rough_actions.append(connect)
            e.used = True
            e.inverse.used = True

        # EMBED actions for the newly output frame.
        for e in rough_action.info.edges:
          if e.used or not e.incoming: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and not nb.output and nb.mention is None:
            embed = RoughAction(Action.EMBED)
            embed.action.role = e.role
            embed.info = nb
            embed.other_info = rough_action.info
            rough_actions.append(embed)
            e.used = True
            e.inverse.used = True

        # ELABORATE actions for the newly output frame.
        for e in rough_action.info.edges:
          if e.used or e.incoming: continue

          nb = frame_info.get(e.neighbor, None)
          if nb is not None and not nb.output and nb.mention is None:
            elaborate = RoughAction(Action.ELABORATE)
            elaborate.action.role = e.role
            elaborate.info = nb
            elaborate.other_info = rough_action.info
            rough_actions.append(elaborate)
            e.used = True
            e.inverse.used = True

        # ASSIGN actions for the newly output frame.
        for e in rough_action.info.edges:
          if not e.used and not e.neighbor.islocal() and not e.incoming:
            assign = RoughAction(Action.ASSIGN)
            assign.info = rough_action.info
            assign.action.role = e.role
            assign.action.label = e.neighbor
            rough_actions.append(assign)
            e.used = True

    return actions


# Action table.
class Actions:
  def __init__(self):
    self.table = []
    self.indices = {}
    self.counts = []
    self.disallowed = []
    self.roles = []
    self.role_indices = {}
    self.max_span_length = None
    self.max_connect_source = None
    self.max_connect_target = None
    self.max_refer_target = None
    self.max_embed_target = None
    self.max_elaborate_source = None
    self.max_action_index = None
    self.stop_index = None
    self.shift_index = None


  def stop(self):
    return self.stop_index


  def shift(self):
    return self.shift_index


  def size(self):
    return len(self.table)


  def max_index(self):
    return self.max_action_index


  def add(self, action):
    index = self.indices.get(action, len(self.table))
    if index == len(self.table):
      self.indices[action] = index
      self.table.append(action)
      self.counts.append(0)
      if action.role is not None and action.role not in self.role_indices:
        role_index = len(self.roles)
        self.role_indices[action.role] = role_index
        self.roles.append(action.role)
    self.counts[index] = self.counts[index] + 1

    if action.type == Action.SHIFT:
      self.shift_index = index
    if action.type == Action.STOP:
      self.stop_index = index


  def _disallow(self, action_type, action_field, percentile):
    values = []
    for action, count in zip(self.table, self.counts):
      if action_type is None or action.type == action_type:
        value = action.__dict__[action_field]
        if value is not None:
          values.extend([value] * count)
    if len(values) == 0: return 0

    values.sort()
    index = int(len(values) * (1.0 * percentile) / 100)
    cutoff = values[index]

    for index, action in enumerate(self.table):
      if action_type is None or action.type == action_type:
        value = action.__dict__[action_field]
        disallowed = value is not None and value > cutoff
        self.disallowed[index] = self.disallowed[index] | disallowed

    return cutoff


  def prune(self, percentile):
    self.disallowed = [False] * self.size()
    self.max_span_length = self._disallow(None, "length", percentile)
    self.max_connect_source = self._disallow(Action.CONNECT, "source", percentile)
    self.max_connect_target = self._disallow(Action.CONNECT, "target", percentile)
    self.max_assign_source = self._disallow(Action.ASSIGN, "source", percentile)
    self.max_refer_target = self._disallow(Action.REFER, "target", percentile)
    self.max_embed_target = self._disallow(Action.EMBED, "target", percentile)
    self.max_elaborate_source = self._disallow(Action.ELABORATE, "source", percentile)
    self.max_action_index = max([
      self.max_connect_source, self.max_connect_target,
      self.max_refer_target, self.max_embed_target,
      self.max_elaborate_source])


# Stores raw feature indices.
class Feature:
  def __init__(self):
    self.indices = []
    self.offsets = []


  def add(self, index):
    self.indices.append(index)


  def new_offset(self):
    assert len(self.offsets) == 0 or self.offsets[-1] < len(self.indices)
    self.offsets.append(len(self.indices))


# Specification for a single link or fixed feature.
class FeatureSpec:
  def __init__(self, name, dim, vocab=None, activation=None, num_links=None):
    self.name = name
    self.dim = dim                     # embedding dimensionality
    self.vocab_size = vocab            # vocabulary size (fixed features only)
    self.activation_size = activation  # activation size (link features only)
    self.num_links = num_links         # no. of links (link features only)


# Training specification.
class Spec:
  def __init__(self):
    # Lexicon generation settings.
    self.words_normalize_digits = True
    self.words_min_count = 1
    self.suffixes_normalize_digits = False
    self.suffixes_min_count = 1
    self.suffixes_max_length = 3

    # Action table percentile.
    self.actions_percentile = 99

    # Network dimensionalities.
    self.lstm_hidden_dim = 256
    self.ff_hidden_dim = 128

    # Fixed feature dimensionalities.
    self.words_dim = 32
    self.suffixes_dim = 16
    self.roles_dim = 16

    # History feature size.
    self.history_limit = 4

    # Frame limit for other link features.
    self.frame_limit = 5

    # Resources.
    self.commons = None
    self.actions = None
    self.words = None
    self.suffix = None


  # Builds an action table from 'corpora'.
  def _build_action_table(self, corpora):
    self.actions = Actions()
    generator = TransitionGenerator(self.commons)
    for document in corpora.documents:
      actions = generator.generate(document)
      for action in actions:
        self.actions.add(action)
        document.gold.append(action)

    self.actions.prune(self.actions_percentile)
    self.num_actions = self.actions.size()
    print self.num_actions, "unique gold actions before pruning"

    allowed = self.num_actions - sum(self.actions.disallowed)
    print "num allowed actions after pruning", allowed
    print len(self.actions.roles), "unique roles in action table"


  # Returns suffix(es) of 'word'.
  def get_suffixes(self, word):
    unicode_chars = list(word.decode("utf-8"))
    output = []
    for length in xrange(1, self.suffixes_max_length + 1):
      if len(unicode_chars) >= length:
        output.append("".join(unicode_chars[-length:]))
    return output


  def ff_fixed(self, features, name, dim, vocab):
    self.ff_fixed_features.append(FeatureSpec(name, dim=dim, vocab=vocab))


  def ff_link(self, features, name, dim, activation, num_links):
    self.ff_link_features.append(
        FeatureSpec(name, dim=dim, activation=activation, num_links=num_links))


  # Builds the spec using 'corpora'.
  def build(self, commons, corpora):
    # Prepare lexical dictionaries.
    self.commons = commons
    self.words = Lexicon(self.words_min_count, self.words_normalize_digits)
    self.suffix =
      Lexicon(self.suffixes_min_count, self.suffixes_normalize_digits)

    for document in corpora.documents:
      for token in document.tokens:
        word = token.text
        self.words.add(word)
        for s in self.get_suffixes(word):
          self.suffix.add(s)
    self.words.finalize()
    self.suffix.finalize()

    # Prepare action table.
    self._build_action_table(corpora)

    # LSTM features.
    self.lstm_features = [
      FeatureSpec("word", dim=self.words_dim, vocab=self.words.size()),
      FeatureSpec("suffix", dim=self.suffixes_dim, vocab=self.suffix.size())
    ]
    self.lstm_input_dim = sum([f.dim for f in self.lstm_features])
    print "LSTM input dim", self.lstm_input_dim
    assert self.lstm_input_dim > 0

    # Feed forward features.
    num_roles = len(self.actions.roles)
    self.ff_fixed_features = []
    self.ff_link_features = []
    fl = self.frame_limit
    if num_roles > 0:
      self.ff_fixed("in_roles", self.roles_dim, num_roles * fl)
      self.ff_fixed("out_roles", self.roles_dim, num_roles * fl)
      self.ff_fixed("unlabeled_roles", self.roles_dim, fl * fl)
      self.ff_fixed("labeled_roles", self.roles_dim, num_roles * fl * fl)

    self.ff_link("frame_creation", 64, self.ff_hidden_dim, fl))
    self.ff_link("frame_focus", 64, self.ff_hidden_dim, fl))
    self.ff_link("frame_end_lr", 64, self.lstm_hidden_dim, fl))
    self.ff_link("frame_end_rl", 64, self.lstm_hidden_dim, fl))
    self.ff_link("lr", 32, self.lstm_hidden_dim, 1))
    self.ff_link("rl", 32, self.lstm_hidden_dim, 1))
    self.ff_link("history", 64, self.ff_hidden_dim, self.history_limit))

    self.ff_input_dim = sum([f.dim for f in self.ff_fixed_features])
    self.ff_input_dim +=
        sum([f.dim * f.num_links for f in self.ff_link_features])
    print "FF input dim", self.ff_input_dim
    assert self.ff_input_dim > 0

    # TODO: Write the following:
    # - Lexicons with their flags (e.g. normalize digits, suffix length)
    # - Action table.
    # - Feature specs.
    #
    # Does it make sense to write all this in one giant encoded SLING frame?


  # Returns raw indices of LSTM features for all tokens in 'document'.
  def raw_lstm_features(self, document):
    output = []
    for f in self.lstm_features:
      features = Feature()
      output.append(features)
      if f.name == "word":
        for token in document.tokens:
          features.new_offset()
          features.add(self.words.index(token.text))
      elif f.name == "suffix":
        for token in document.tokens:
          features.new_offset()
          for s in self.get_suffixes(token.text):
            features.add(self.suffix.index(s))
      else:
        raise ValueError("LSTM feature '", f.name, "' not implemented")
    return output


  # Returns raw indices of all fixed FF features for 'state'.
  def raw_ff_fixed_features(self, feature_spec, state):
    role_graph = state.role_graph()
    num_roles = len(self.actions.roles)
    fl = self.frame_limit
    raw_features = []
    if feature_spec.name == "in_roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[2] * num_roles + e[1])
    elif feature_spec.name == "out_roles":
      for e in role_graph:
        raw_features.append(e[0] * num_roles + e[1])
    elif feature_spec.name == "unlabeled_roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[0] * fl + e[2])
    elif feature_spec.name == "labeled_roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[0] * fl * num_roles + e[2] * num_roles + e[1])
    else:
      raise ValueError("FF feature '", feature_spec.name, "' not implemented")

    return raw_features


  def translated_ff_link_features(self, feature_spec, state):
    name = feature_spec.name
    num = features_spec.num_links

    output = []
    if name == "history":
      for i in xrange(num):
        output.append(None if i < state.steps else -1 - i)
    elif name == "lr":
      index = None
      if state.current < state.end:
        index = state.current - state.begin
      output.append(index)
    elif name == "rl":
      index = None
      if state.current < state.end:
        index = -1 - (state.current - state.begin)
      output.append(index)
    elif name == "frame_end_lr":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = end - state.begin
        output.append(index)
    elif name == "frame_end_rl":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = -1 - (end - state.begin)
        output.append(index)
    elif name == "frame_creation":
      for i in xrange(num):
        step = state.creation_step(i)
        output.append(None if step == -1 else step)
    elif name == "frame_focus":
      for i in xrange(num):
        step = state.focus_step(i)
        output.append(None if step == -1 else step)
    else:
      raise ValueError("Link feature not implemented:" + name)

    return output


  def lstm_feature_strings(self, feature_spec, indices):
    strings = []
    if feature_spec.name == "word":
      strings = [self.words.value(index) for index in indices]
    elif feature_spec.name == "suffix":
      strings = [self.suffix.value(index) for index in indices]
    else:
      raise ValueError(feature_spec.name + " not implemented")
    return str(strings)


  def ff_fixed_feature_strings(self, feature_spec, indices):
    limit = self.frame_limit
    roles = self.actions.roles
    nr = len(roles)

    strings = []
    if feature_spec.name == "out_roles":
      strings = [str(i / nr) + "->" + str(roles[i % nr]) for i in indices]
    elif feature_spec.name == "in_roles":
      strings = [str(roles[i % nr]) + "->" + str(i / nr) for i in indices]
    elif feature_spec.name == "unlabeled_roles":
      strings = [str(i / limit) + "->" + str(i % limit) for i in indices]
    elif feature_spec.name == "labeled_roles":
      t = limit * nr
      for i in indices:
        value = str(i / t) + "->" + str(roles[(i % t) % nr])
        value += "->" + str((i % t) / nr)
        strings.append(value)
    else:
      raise ValueError(feature_spec.name + " not implemented")
    return str(strings)


class ParserState:
  class Span:
    def __init__(self, start, length):
      self.start = start
      self.end = start + length
      self.evoked = []


  class Frame:
    def __init__(self, t):
      self.type = t
      self.edges = []
      self.start = -1
      self.end = -1
      self.focus = 0
      self.creation = 0
      self.spans = []


  def __init__(self, document, spec):
    self.document = document
    self.spec = spec
    self.current = 0
    self.begin = 0
    self.end = len(document.tokens)
    self.frames = []
    self.spans = []
    self.steps = 0
    self.graph = []
    self.allowed = [False] * spec.actions.size()
    self.done = False
    self.attention = []
    self.nesting = []
    self.embed = []
    self.elaborate = []
    self.actions = []


  def __repr__(self):
    s = "Curr:" + str(self.current) + " in [" + str(self.begin) +
        ", " + str(self.end) + ")" + " " + str(len(self.frames)) + " frames"
    for index, f in enumerate(self.attention):
      if index == 10: break
      s += "\n   - Attn " + str(index) + ":" + str(f.type) +
           " Creation:" + str(f.creation) +
           ", Focus:" + str(f.focus) + ", #Edges:" + str(len(f.edges)) +
           " (" + str(len(f.spans)) + " spans) "
      if len(f.spans) > 0:
        for span in f.spans:
          words = self.document.tokens[span.start].text
          if span.end > span.start + 1:
            words += ".." + self.document.tokens[span.end - 1].text
          s += words + " = [" + str(span.start) + ", " + str(span.end) + ") "
    return s


  def compute_role_graph(self):
    if len(self.spec.actions.roles) == 0: return
    del self.graph
    self.graph = []
    limit = min(self.spec.frame_limit, len(self.attention))
    for i in xrange(limit):
      frame = self.attention[i]
      for role, value in frame.edges:
        role_id = self.spec.actions.role_indices.get(role, None)
        if role_id is not None:
          self.graph.append((i, role_id, self.index(value)))


  def creation_step(self, index):
    if index >= len(self.attention) or index < 0: return -1
    return self.attention[index].creation


  def focus_step(self, index):
    if index >= len(self.attention) or index < 0: return -1
    return self.attention[index].focus


  def is_allowed(self, action_index):
    if self.done: return False
    if self.current == self.end: return action_index == self.spec.actions.stop()
    if action_index == self.spec.actions.stop(): return False
    if action_index == self.spec.actions.shift(): return True
    action = self.spec.actions.table[action_index]

    if action.type == Action.EVOKE or action.type == Action.REFER:
      if self.current + action.length > self.end: return False
      if action.type == Action.REFER and action.target >= self.attention_size():
        return False
      if len(self.nesting) == 0: return True

      outer = self.nesting[-1]
      assert outer.start <= self.current
      assert outer.end > self.current
      gap = outer.end - self.current
      if gap > action.length:
        return True
      elif gap < action.length:
        return False
      elif outer.start < self.current:
        return True
      else:
        if action.type == Action.EVOKE:
          for f in outer.evoked:
            if f.type == action.label: return False
          return True
        else:
          target = self.attention[action.target]
          for f in outer.evoked:
            if f is target: return False
          return True
    elif action.type == Action.CONNECT:
      s = self.attention_size()
      if action.source >= s or action.target >= s: return False
      source = self.attention[action.source]
      target = self.attention[action.target]
      for role, value in source.edges:
        if role == action.role and value is target: return False
      return True
    elif action.type == Action.EMBED:
      if action.target >= self.attention_size(): return False
      target = self.attention[action.target]
      for t, role, value in self.embed:
        if t == action.label and role == action.role and value is target:
          return False
      return True
    elif action.type == Action.ELABORATE:
      if action.source >= self.attention_size(): return False
      source = self.attention[action.source]
      for t, role, value in self.elaborate:
        if t == action.label and role == action.role and value is source:
          return False
      return True
    elif action.type == Action.ASSIGN:
      if action.source >= self.attention_size(): return False
      source = self.attention[action.source]
      for role, value in source.edges:
        if role == action.role and value == action.label: return False
      return True
    else:
      raise ValueError("Unknown action : ", action)


  def index(self, frame):
    for i in xrange(len(self.attention)):
      if self.attention[i] is frame:
        return i
    return -1


  def frame(self, index):
    return self.attention[index]


  def attention_size(self):
    return len(self.attention)


  def role_graph(self):
    return self.graph


  def _add_to_attention(self, f):
    f.focus = self.steps
    self.attention.insert(0, f)


  def _refocus_attention(self, index):
    f = self.attention[index]
    f.focus = self.steps
    if index > 0: self.attention.insert(0, self.attention.pop(index))


  def _make_span(self, length):
    # See if an existing span can be returned.
    if len(self.nesting) > 0:
      last = self.nesting[-1]
      if last.start == self.current and last.end == self.current + length:
        return last
    s = ParserState.Span(self.current, length)
    self.spans.append(s)
    self.nesting.append(s)
    return s


  def advance(self, action):
    self.actions.append(action)
    if action.type == Action.STOP:
      self.done = True
    elif action.type == Action.SHIFT:
      self.current += 1
      while len(self.nesting) > 0:
        if self.nesting[-1].end <= self.current:
          self.nesting.pop()
        else:
          break
      del self.embed[:]
      del self.elaborate[:]
    elif action.type == Action.EVOKE:
      s = self._make_span(action.length)
      f = ParserState.Frame(action.label)
      f.start = self.current
      f.end = self.current + action.length
      f.creation = self.steps
      f.spans.append(s)
      s.evoked.append(f)
      self.frames.append(f)
      self._add_to_attention(f)
    elif action.type == Action.REFER:
      f = self.attention[action.target]
      f.focus = self.steps
      s = self._make_span(action.length)
      s.evoked.append(f)
      f.spans.append(s)
      self._refocus_attention(action.target)
    elif action.type == Action.CONNECT:
      f = self.attention[action.source]
      f.edges.append((action.role, self.attention[action.target]))
      f.focus = self.steps
      self._refocus_attention(action.source)
    elif action.type == Action.EMBED:
      target = self.attention[action.target]
      f = ParserState.Frame(action.label)
      f.creation = self.steps
      f.focus = self.steps
      f.edges.append((action.role, target))
      self._add_to_attention(f)
      self.embed.append((action.label, action.role, target))
    elif action.type == Action.ELABORATE:
      source = self.attention[action.source]
      f = ParserState.Frame(action.label)
      f.creation = self.steps
      f.focus = self.steps
      source.edges.append((action.role, f))
      self._add_to_attention(f)
      self.elaborate.append((action.label, action.role, source))
    elif action.type == Action.ASSIGN:
      source = self.attention[action.source]
      source.focus = self.steps
      source.edges.append((action.role, action.label))
      self._refocus_attention(action.source)
    else:
      raise ValueError("Unknown action type: ", action.type)

    self.steps += 1
    if action.type != Action.SHIFT and action.type != Action.STOP:
      self.compute_role_graph()


  def frame_end_inclusive(self, index):
    if index >= len(self.attention) or index < 0:
      return -1
    else:
      return self.attention[index].end - 1



