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

import os
import psutil
import random
import sling
import struct
import subprocess
import unicodedata

from datetime import datetime


# Class for computing and serving a lexicon.
# Usage:
#   lexicon = Lexicon(normalize_digits=...)
#   lexicon.add("foo")
#   lexicon.add("bar")
#   ...
#   index = lexicon.index("foo")
class Lexicon:
  def __init__(self, normalize_digits=True, oov_item="<UNKNOWN>"):
    self.normalize_digits = normalize_digits
    self.item_to_index = {}
    self.index_to_item = {}

    if oov_item is not None:
      self.oov_item = oov_item
      self.oov_index = 0   # Don't change this; OOV is always at position 0
      self.index_to_item[self.oov_index] = self.oov_item
      self.item_to_index[self.oov_item] = self.oov_index
    else:
      self.oov_item = None
      self.oov_index = None


  def has_oov(self):
    return self.oov_index is not None


  def _key(self, item):
    if self.normalize_digits:
      return "".join([c if not c.isdigit() else '9' for c in list(item)])
    else:
      return item


  def load(self, vocabfile):
    with open(vocabfile, "r") as f:
      index = 0
      for line in f:
        line = line.strip()
        if line == self.oov_item:
          assert index == self.oov_index, index
        self.item_to_index[line] = index
        self.index_to_item[index] = line
        index += 1


  def add(self, item):
    item = self._key(item)
    if item not in self.item_to_index:
      i = len(self.item_to_index)
      self.item_to_index[item] = i
      self.index_to_item[i] = item


  def size(self):
    return len(self.item_to_index)


  def index(self, item):
    item = self._key(item)
    if item not in self.item_to_index:
      return self.oov_index  # None if !has_oov()
    else:
      return self.item_to_index[item]


  def value(self, index):
    assert index >= 0 and index < len(self.index_to_item), "%r" % index
    return self.index_to_item[index]


  def __str__(self):
    s = [self.index_to_item[i] for i in xrange(self.size())]
    return "\n".join(s)


  def first_few(self, prefix="", n=100):
    s = []
    for i in xrange(min(n, self.size())):
      s.append(prefix + str(i) + " = " + self.index_to_item[i])
    return "\n".join(s)


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
    self.inner.mentions.sort(cmp=mention_comparator)
    self.gold = []  # sequence of gold transitions


  def mentions(self):
    return self.inner.mentions


  def themes(self):
    return self.inner.themes


  def tokens(self):
    return self.inner.tokens


  def size(self):
    return len(self.tokens())


# An iterator over a recordio of documents. It doesn't load all documents into
# memory at once, and so can't shuffle the corpus. It can optionally loop over
# the corpus, and compute transition sequences for the existing frames in the
# document.
class Corpora:
  def __init__(self, recordio, commons, schema, gold=False, loop=False):
    self.filename = recordio
    self.commons = commons
    self.schema = schema
    self.reader = sling.RecordReader(recordio)
    self.generator = None
    self.loop = loop
    self.generator = None
    self.set_gold(gold)


  def __del__(self):
    self.reader.close()


  def __iter__(self):
    return self


  def set_loop(self, val):
    self.loop = val


  def set_gold(self, gold):
    if gold and self.generator is None:
      self.generator = TransitionGenerator(self.commons)
    elif not gold:
      self.generator = None


  def next(self):
    if self.reader.done():
      if self.loop:
        self.reader.rewind()
      else:
        raise StopIteration

    (_, value) = self.reader.next()
    document = Document(self.commons, self.schema, value)
    if self.generator is not None:
      document.gold = self.generator.generate(document)
    return document


  def rewind(self):
    self.reader.rewind()


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
      info = TransitionGenerator.FrameInfo(frame)
      frame_info[frame] = info

    pending = []
    for role, value in frame:
      if not self.is_ref(value) or role == self._id \
        or (role == self._isa and value.islocal()):
        continue
      if role == self._isa and info.type is None:
        info.type = value
      else:
        edge = TransitionGenerator.Edge(incoming=False, role=role, value=value)
        info.edges.append(edge)
        if value == frame:
          edge.inverse = edge
          continue

        if value.islocal():
          nb_info = frame_info.get(value, None)
          if nb_info is None:
            nb_info = TransitionGenerator.FrameInfo(value)
            frame_info[value] = nb_info
          nb_edge = TransitionGenerator.Edge(
              incoming=True, role=role, value=frame)
          nb_info.edges.append(nb_edge)
          nb_edge.inverse = edge
          edge.inverse = nb_edge
          pending.append(value)

    # Assign a fallback type.
    if info.type is None:
      info.type = self._thing_type

    # Initialize bookkeeping for all frames pointed to by this frame.
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


  def _rough_action(self, type=None):
    return TransitionGenerator.RoughAction(type)


  def generate(self, document):
    frame_info = {}
    initialized = {}
    for m in document.mentions():
      for evoked in m.evokes():
        self._init_info(evoked, frame_info, initialized)
        frame_info[evoked].mention = m

    for theme in document.themes():
      self._init_info(theme, frame_info, initialized)

    rough_actions = []
    start = 0
    evoked = {}
    for m in document.mentions():
      for i in xrange(start, m.begin):
        rough_actions.append(self._rough_action(Action.SHIFT))
      start = m.begin

      for frame in m.evokes():
        rough_action = self._rough_action()
        rough_action.action.length = m.length
        rough_action.info = frame_info[frame]
        if frame not in evoked:
          rough_action.action.type = Action.EVOKE
          evoked[frame] = True
        else:
          rough_action.action.type = Action.REFER
        rough_actions.append(rough_action)

    for index in xrange(start, document.size()):
      rough_actions.append(self._rough_action(Action.SHIFT))

    rough_actions.append(self._rough_action(Action.STOP))
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
            connect = self._rough_action(Action.CONNECT)
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
            embed = self._rough_action(Action.EMBED)
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
            elaborate = self._rough_action(Action.ELABORATE)
            elaborate.action.role = e.role
            elaborate.info = nb
            elaborate.other_info = rough_action.info
            rough_actions.append(elaborate)
            e.used = True
            e.inverse.used = True

        # ASSIGN actions for the newly output frame.
        for e in rough_action.info.edges:
          if not e.used and not e.neighbor.islocal() and not e.incoming:
            assign = self._rough_action(Action.ASSIGN)
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
    self.stop_index = None
    self.shift_index = None


  def stop(self):
    return self.stop_index


  def shift(self):
    return self.shift_index


  def size(self):
    return len(self.table)


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


  def _maxvalue(self, action_type, action_field, percentile):
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
    p = percentile
    if p < 1 and type(p) is float: p = int(p * 100)

    self.disallowed = [False] * self.size()
    self.max_span_length = self._maxvalue(None, "length", percentile)
    self.max_connect_source = self._maxvalue(Action.CONNECT, "source", p)
    self.max_connect_target = self._maxvalue(Action.CONNECT, "target", p)
    self.max_assign_source = self._maxvalue(Action.ASSIGN, "source", p)
    self.max_refer_target = self._maxvalue(Action.REFER, "target", p)
    self.max_embed_target = self._maxvalue(Action.EMBED, "target", p)
    self.max_elaborate_source = self._maxvalue(Action.ELABORATE, "source", p)


  # Encodes and returns the action table as a SLING frame.
  def encoded(self, commons):
    store = sling.Store(commons)
    table = store.frame({"id": "/table"})

    table["/table/max_span_length"] = self.max_span_length
    table["/table/max_connect_source"] = self.max_connect_source
    table["/table/max_connect_target"] = self.max_connect_target
    table["/table/max_assign_source"] = self.max_assign_source
    table["/table/max_refer_target"] = self.max_refer_target
    table["/table/max_embed_target"] = self.max_embed_target
    table["/table/max_elaborate_source"] = self.max_elaborate_source

    def fill(f, name, val):
      if val is not None:
        f["/table/action/" + name] = val

    actions_array = store.array(self.size())
    for index, action in enumerate(self.table):
      frame = store.frame({})
      fill(frame, "type", action.type)
      fill(frame, "length", action.length)
      fill(frame, "source", action.source)
      fill(frame, "target", action.target)
      fill(frame, "label", action.label)
      fill(frame, "role", action.role)
      actions_array[index] = frame
    table["/table/actions"] = actions_array

    symbols = map(
        lambda i: "/table/action/" + i,
        ["type", "length", "source", "target", "role", "label"])
    table["/table/symbols"] = symbols
    return table.data(binary=True)


  def __str__(self):
    s = ["Action %d = %s" % (i, a) for i, a in enumerate(self.table)]
    return "\n".join(s)


# Stores raw feature indices.
class Feature:
  def __init__(self):
    self.indices = []       # list of lists of indices
    self.has_empty = False  # are some of the lists in 'indices' empty
    self.has_multi = False  # do some of the lists have >1 values


  # Adds 'index' to 'indices'. 'index' could be one index or a list of indices.
  def add(self, index):
    if type(index) is int:
      self.indices.append(index)
    else:
      assert type(index) is list
      if len(index) == 1:
        self.indices.append(index[0])
      else:
        self.indices.append(index)
        if len(index) == 0: self.has_empty = True
        if len(index) > 1: self.has_multi = True


# Specification for a single link or fixed feature.
class FeatureSpec:
  def __init__(self, name, dim, vocab=None, activation=None, num=1):
    self.name = name
    self.dim = dim                     # embedding dimensionality
    self.vocab_size = vocab            # vocabulary size (fixed features only)
    self.activation_size = activation  # activation size (link features only)
    self.num = num                     # no. of links / no. of fixed feature ids


# Training specification.
class Spec:
  # Fallback feature values.
  NO_HYPHEN = 0
  HAS_HYPHEN = 1
  HYPHEN_CARDINALITY = 2

  LOWERCASE = 0
  UPPERCASE = 1
  CAPITALIZED = 2
  INITIAL = 3
  NON_ALPHABETIC = 4
  CAPITALIZATION_CARDINALITY = 5

  NO_PUNCTUATION = 0
  SOME_PUNCTUATION = 1
  ALL_PUNCTUATION = 2
  PUNCTUATION_CARDINALITY = 3

  NO_QUOTE = 0
  OPEN_QUOTE = 1
  CLOSE_QUOTE = 2
  UNKNOWN_QUOTE = 3
  QUOTE_CARDINALITY = 4

  NO_DIGIT = 0
  SOME_DIGIT = 1
  ALL_DIGIT = 2
  DIGIT_CARDINALITY = 3

  def __init__(self):
    # Lexicon generation settings.
    self.words_normalize_digits = True
    self.suffixes_normalize_digits = False
    self.suffixes_max_length = 3

    # Action table percentile.
    self.actions_percentile = 99

    # Network dimensionalities.
    self.lstm_hidden_dim = 256
    self.ff_hidden_dim = 128

    # Fixed feature dimensionalities.
    self.oov_features = True
    self.words_dim = 32
    self.suffixes_dim = 16
    self.fallback_dim = 8  # dimensionality of each fallback feature
    self.roles_dim = 16

    # History feature size.
    self.history_limit = 4

    # Frame limit for other link features.
    self.frame_limit = 5

    # Resources.
    self.commons = None
    self.commons_path = None
    self.actions = None
    self.words = None
    self.suffix = None
    self.word_embeddings = None
    self.word_embedding_indices = None

    # To be determined.
    self.num_actions = None
    self.lstm_features = []
    self.ff_fixed_features = []
    self.ff_link_features = []


  # Builds an action table from 'corpora'.
  def _build_action_table(self, corpora):
    corpora.rewind()
    self.actions = Actions()
    for document in corpora:
      assert document.size() == 0 or len(document.gold) > 0
      for action in document.gold:
        self.actions.add(action)

    self.actions.prune(self.actions_percentile)
    self.num_actions = self.actions.size()
    print self.num_actions, "unique gold actions before pruning"

    allowed = self.num_actions - sum(self.actions.disallowed)
    print "num allowed actions after pruning", allowed
    print len(self.actions.roles), "unique roles in action table"


  # Returns suffix(es) of 'word'.
  def get_suffixes(self, word, unicode_chars=None):
    if unicode_chars is None:
      unicode_chars = list(word.decode("utf-8"))
    output = []
    end = min(self.suffixes_max_length, len(unicode_chars))
    for start in xrange(end, 0, -1):
      output.append("".join(unicode_chars[-start:]))
    return output


  # Dumps suffixes in the AffixTable format.
  # See sling/nlp/document/affix.cc for details.
  def dump_suffixes(self, buf=None):
    if buf is None: buf = bytearray()

    # Writes 'num' in varint encoding to 'b'.
    def writeint(num, b):
      while True:
        part = num & 127
        num = num >> 7
        if num > 0:
          b.append(part | 128)
        else:
          b.append(part)
          break


    writeint(1, buf)  # 1 = AffixTable::SUFFIX
    writeint(self.suffixes_max_length, buf)
    writeint(self.suffix.size(), buf)
    for i in xrange(self.suffix.size()):
      v = self.suffix.value(i)
      assert type(v) is unicode
      v_str = v.encode("utf-8")

      writeint(len(v_str), buf)       # number of bytes
      for x in v_str: buf.append(x)   # the bytes themselves
      writeint(len(v), buf)           # number of characters
      if len(v) > 1:
        shorter = v[1:]
        shorter_idx = self.suffix.index(shorter)
        assert shorter_idx is not None, shorter
        writeint(shorter_idx, buf)    # id of the suffix one character shorter

    return buf


  # Adds LSTM feature to the specification.
  def add_lstm_fixed(self, name, dim, vocab, num=1):
    self.lstm_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))

  # Adds fixed feature to the specification.
  def add_ff_fixed(self, name, dim, vocab, num):
    self.ff_fixed_features.append(
        FeatureSpec(name, dim=dim, vocab=vocab, num=num))


  # Adds link feature to the specification.
  def add_ff_link(self, name, dim, activation, num):
    self.ff_link_features.append(
        FeatureSpec(name, dim=dim, activation=activation, num=num))


  # Specifies all fixed and link features.
  def _specify_features(self):
    # LSTM features.
    self.add_lstm_fixed("words", self.words_dim, self.words.size())
    if self.oov_features:
      self.add_lstm_fixed(
          "suffix", self.suffixes_dim, self.suffix.size(), \
          self.suffixes_max_length)
      self.add_lstm_fixed(
          "capitalization", self.fallback_dim, Spec.CAPITALIZATION_CARDINALITY)
      self.add_lstm_fixed("hyphen", self.fallback_dim, Spec.HYPHEN_CARDINALITY)
      self.add_lstm_fixed(
          "punctuation", self.fallback_dim, Spec.PUNCTUATION_CARDINALITY)
      self.add_lstm_fixed("quote", self.fallback_dim, Spec.QUOTE_CARDINALITY)
      self.add_lstm_fixed("digit", self.fallback_dim, Spec.DIGIT_CARDINALITY)

    self.lstm_input_dim = sum([f.dim for f in self.lstm_features])
    print "LSTM input dim", self.lstm_input_dim
    assert self.lstm_input_dim > 0

    # Feed forward features.
    num_roles = len(self.actions.roles)
    fl = self.frame_limit
    if num_roles > 0:
      num = 32
      dim = self.roles_dim
      self.add_ff_fixed("in-roles", dim, num_roles * fl, num)
      self.add_ff_fixed("out-roles", dim, num_roles * fl, num)
      self.add_ff_fixed("labeled-roles", dim, num_roles * fl * fl, num)
      self.add_ff_fixed("unlabeled-roles", dim, fl * fl, num)

    self.add_ff_link("frame-creation-steps", 64, self.ff_hidden_dim, fl)
    self.add_ff_link("frame-focus-steps", 64, self.ff_hidden_dim, fl)
    self.add_ff_link("frame-end-lr", 32, self.lstm_hidden_dim, fl)
    self.add_ff_link("frame-end-rl", 32, self.lstm_hidden_dim, fl)
    self.add_ff_link("history", 64, self.ff_hidden_dim, self.history_limit)
    self.add_ff_link("lr", 32, self.lstm_hidden_dim, 1)
    self.add_ff_link("rl", 32, self.lstm_hidden_dim, 1)

    self.ff_input_dim = sum([f.dim for f in self.ff_fixed_features])
    self.ff_input_dim += sum(
        [f.dim * f.num for f in self.ff_link_features])
    print "FF_input_dim", self.ff_input_dim
    assert self.ff_input_dim > 0


  # Builds the spec using 'corpora'.
  def build(self, commons, corpora):
    if type(commons) is str:
      self.commons_path = commons
      commons = sling.Store()
      commons.load(self.commons_path)
      commons.freeze()

    # Prepare lexical dictionaries.
    # For compatibility with DRAGNN, suffixes don't have an OOV item.
    self.commons = commons
    self.words = Lexicon(self.words_normalize_digits)
    self.suffix = Lexicon(self.suffixes_normalize_digits, oov_item=None)

    corpora.rewind()
    corpora.set_gold(False)   # No need to compute gold transitions yet
    for document in corpora:
      for token in document.tokens():
        word = token.text
        self.words.add(word)
        for s in self.get_suffixes(word):
          self.suffix.add(s)
    print "Words:", self.words.size(), "items in lexicon, including OOV"
    print "Suffix:", self.suffix.size(), "items in lexicon"

    # Prepare action table.
    corpora.set_gold(True)
    self._build_action_table(corpora)

    # Add feature specs.
    self._specify_features()


  # Loads embeddings for words in the lexicon.
  def load_word_embeddings(self, embeddings_file):
    self.word_embeddings = [None] * self.words.size()
    f = open(embeddings_file, 'rb')

    # Read header.
    header = f.readline().strip()
    size = int(header.split()[0])
    dim = int(header.split()[1])
    assert dim == self.words_dim, "%r vs %r" % (dim, self.words_dim)

    # Read vectors for known words.
    count = 0
    fmt = "f" * dim
    vector_size = 4 * dim  # 4 being sizeof(float)
    oov = self.words.oov_index
    for _ in xrange(size):
      word = ""
      while True:
        ch = f.read(1)
        if ch == " ": break
        word += ch

      vector = list(struct.unpack(fmt, f.read(vector_size)))
      ch = f.read(1)
      assert ch == "\n", "%r" % ch     # end of line expected

      index = self.words.index(word)
      if index != oov and self.word_embeddings[index] is None:
        self.word_embeddings[index] = vector
        count += 1

    f.close()

    self.word_embedding_indices =\
        [i for i, v in enumerate(self.word_embeddings) if v is not None]
    self.word_embeddings = [v for v in self.word_embeddings if v is not None]

    print "Loaded", count, "pre-trained embeddings from file with", size, \
        "vectors. Vectors for remaining", (self.words.size() - count), \
        "words will be randomly initialized."


  # Returns raw indices of LSTM features for all tokens in 'document'.
  def raw_lstm_features(self, document):
    output = []
    chars = []
    categories = []
    for token in document.tokens():
      decoding = list(token.text.decode("utf-8"))
      chars.append(decoding)
      categories.append([unicodedata.category(ch) for ch in decoding])

    for f in self.lstm_features:
      features = Feature()
      output.append(features)
      if f.name == "words":
        for token in document.tokens():
          features.add(self.words.index(token.text))
      elif f.name == "suffix":
        for index, token in enumerate(document.tokens()):
          suffixes = self.get_suffixes(token.text, chars[index])
          ids = [self.suffix.index(s) for s in suffixes]
          features.add([i for i in ids if i is not None])  # no OOV in suffixes
      elif f.name == "hyphen":
        for index, token in enumerate(document.tokens()):
          hyphen = any(c == 'Pd' for c in categories[index])
          features.add(Spec.HAS_HYPHEN if hyphen else Spec.NO_HYPHEN)
      elif f.name == "capitalization":
        for index, token in enumerate(document.tokens()):
          has_upper = any(c == 'Lu' for c in categories[index])
          has_lower = any(c == 'Ll' for c in categories[index])

          value = Spec.CAPITALIZED
          if not has_upper and has_lower:
            value = Spec.LOWERCASE
          elif has_upper and not has_lower:
            value = Spec.UPPERCASE
          elif not has_upper and not has_lower:
            value = Spec.NON_ALPHABETIC
          elif index == 0 or token.brk >= 3:  # 3 = SENTENCE_BREAK
            value = Spec.INITIAL
          features.add(value)
      elif f.name == "punctuation":
        for index in xrange(len(document.tokens())):
          all_punct = all(c[0] == 'P' for c in categories[index])
          some_punct = any(c[0] == 'P' for c in categories[index])

          if all_punct:
            features.add(Spec.ALL_PUNCTUATION)
          elif some_punct:
            features.add(Spec.SOME_PUNCTUATION)
          else:
            features.add(Spec.NO_PUNCTUATION)

      elif f.name == "digit":
        for index in xrange(len(document.tokens())):
          all_digit = all(c == 'Nd' for c in categories[index])
          some_digit = any(c == 'Nd' for c in categories[index])

          if all_digit:
            features.add(Spec.ALL_DIGIT)
          elif some_digit:
            features.add(Spec.SOME_DIGIT)
          else:
            features.add(Spec.NO_DIGIT)

      elif f.name == "quote":
        in_quote = False
        for index in xrange(len(document.tokens())):
          value = Spec.NO_QUOTE
          for cat, ch in zip(categories[index], chars[index]):
            if cat == 'Pi':
              value = Spec.OPEN_QUOTE
            elif cat == 'Pf':
              value = Spec.CLOSE_QUOTE
            elif cat == 'Po' and (ch == '\'' or ch == '"'):
              value = Spec.UNKNOWN_QUOTE
            elif cat == 'Sk' and ch == '`':
              value = Spec.UNKNOWN_QUOTE
          if value != Spec.NO_QUOTE:
            token = document.tokens()[index]
            if token.text == "``":
              value = Spec.OPEN_QUOTE
            elif token.text == "''":
              value = Spec.CLOSE_QUOTE
            if value == Spec.UNKNOWN_QUOTE:
              value = Spec.CLOSE_QUOTE if in_quote else Spec.OPEN_QUOTE
              in_quote = not in_quote
          features.add(value)
      else:
        raise ValueError("LSTM feature '", f.name, "' not implemented")
    return output


  # Returns raw indices of all fixed FF features for 'state'.
  def raw_ff_fixed_features(self, feature_spec, state):
    role_graph = state.role_graph()
    num_roles = len(self.actions.roles)
    fl = self.frame_limit
    raw_features = []
    if feature_spec.name == "in-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[2] * num_roles + e[1])
    elif feature_spec.name == "out-roles":
      for e in role_graph:
        raw_features.append(e[0] * num_roles + e[1])
    elif feature_spec.name == "unlabeled-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[2] * fl + e[0])
    elif feature_spec.name == "labeled-roles":
      for e in role_graph:
        if e[2] is not None and e[2] < fl and e[2] >= 0:
          raw_features.append(e[0] * fl * num_roles + e[2] * num_roles + e[1])
    else:
      raise ValueError("FF feature '", feature_spec.name, "' not implemented")

    return raw_features


  def translated_ff_link_features(self, feature_spec, state):
    name = feature_spec.name
    num = feature_spec.num

    output = []
    if name == "history":
      for i in xrange(num):
        output.append(None if i >= state.steps else -1 - i)
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
    elif name == "frame-end-lr":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = end - state.begin
        output.append(index)
    elif name == "frame-end-rl":
      for i in xrange(num):
        index = None
        end = state.frame_end_inclusive(i)
        if end != -1:
          index = -1 - (end - state.begin)
        output.append(index)
    elif name == "frame-creation-steps":
      for i in xrange(num):
        step = state.creation_step(i)
        output.append(None if step == -1 else step)
    elif name == "frame-focus-steps":
      for i in xrange(num):
        step = state.focus_step(i)
        output.append(None if step == -1 else step)
    else:
      raise ValueError("Link feature not implemented:" + name)

    return output


  # Dumps resources as flow blobs.
  def dump_flow(self, flow):
    lexicon = flow.blob("lexicon")
    lexicon.type = "dict"
    lexicon.add_attr("delimiter", 10)
    lexicon.add_attr("oov", self.words.oov_index)
    lexicon.add_attr("normalize_digits", self.words.normalize_digits)
    lexicon.data = str(self.words) + "\n"

    def read_file(filename):
      fin = open(filename, "r")
      data = fin.read()
      fin.close()
      return data

    commons = flow.blob("commons")
    commons.type = "frames"
    commons.data = read_file(self.commons_path)

    suffix = flow.blob("suffixes")
    suffix.type = "affix"
    suffix.data = str(self.dump_suffixes())

    actions = flow.blob("actions")
    actions.type = "frames"
    actions.data = self.actions.encoded(self.commons)


  # Debugging methods.
  #
  # Returns feature strings for LSTM feature indices in 'indices'. All indices
  # are assumed to belong to a single feature whose spec is in 'feature_spec'.
  def lstm_feature_strings(self, feature_spec, indices):
    strings = []
    if feature_spec.name == "words":
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
    if feature_spec.name == "out-roles":
      strings = [str(i / nr) + "->" + str(roles[i % nr]) for i in indices]
    elif feature_spec.name == "in-roles":
      strings = [str(roles[i % nr]) + "->" + str(i / nr) for i in indices]
    elif feature_spec.name == "unlabeled-roles":
      strings = [str(i / limit) + "->" + str(i % limit) for i in indices]
    elif feature_spec.name == "labeled-roles":
      t = limit * nr
      for i in indices:
        value = str(i / t) + "->" + str(roles[(i % t) % nr])
        value += "->" + str((i % t) / nr)
        strings.append(value)
    else:
      raise ValueError(feature_spec.name + " not implemented")
    return str(strings)


  def oracle_trace(self, document):
    assert len(document.gold) > 0, "No gold actions"
    state = ParserState(document, self)
    for gold in document.gold:
      print "Taking gold action", gold
      print "On state:", state

      gold_index = self.actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action: %r" % gold
      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
      state.advance(gold)

    print "Final state after", len(document.gold), "actions:", state


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
    self.end = len(document.tokens())
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
    s = "Curr:" + str(self.current) + " in [" + str(self.begin) + \
        ", " + str(self.end) + ")" + " " + str(len(self.frames)) + " frames"
    for index, f in enumerate(self.attention):
      if index == 10: break
      s += "\n   - Attn " + str(index) + ":" + str(f.type) + \
           " Creation:" + str(f.creation) + \
           ", Focus:" + str(f.focus) + ", #Edges:" + str(len(f.edges)) + \
           " (" + str(len(f.spans)) + " spans) "
      if len(f.spans) > 0:
        for span in f.spans:
          words = self.document.tokens()[span.start].text
          if span.end > span.start + 1:
            words += ".." + self.document.tokens()[span.end - 1].text
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
          target = -1
          if isinstance(value, ParserState.Frame):
            target = self.index(value)
            if target == -1 or target >= self.spec.frame_limit: continue
          self.graph.append((i, role_id, target))


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


  def frame_end_inclusive(self, index):
    if index >= len(self.attention) or index < 0:
      return -1
    else:
      return self.attention[index].end - 1


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


  def write(self, document=None):
    if document is None:
      document = self.document.inner

    assert type(document) == sling.Document

    store = document.frame.store()
    document.remove_annotations()
    frames = {}

    for f in self.frames:
      frame = store.frame({"isa": f.type})
      frames[f] = frame
      if len(f.spans) == 0:
        document.add_theme(frame)

    for f in self.frames:
      frame = frames[f]
      for role, value in f.edges:
        if isinstance(value, ParserState.Frame):
          assert value in frames
          frame.append(role, frames[value])
        else:
          assert type(value) == sling.Frame, "%r" % value
          frame.append(role, value)

    for s in self.spans:
      # Note: mention.frame is the actual mention frame.
      mention = document.add_mention(s.start, s.end)
      for f in s.evoked:
        assert f in frames
        mention.frame.append("/s/phrase/evokes", frames[f])

    document.update()


  def data(self, **kwargs):
    return self.document.inner.frame.data(**kwargs)


  def encoded(self):
    return self.data(binary=True, shallow=True)


  def textual(self):
    return self.data(binary=False, pretty=True, shallow=True)


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


def frame_evaluation(gold_corpus_path, test_corpus_path, commons_path):
  try:
    output = subprocess.check_output(
        ['bazel-bin/sling/nlp/parser/tools/evaluate-frames',
         '--gold_documents=' + gold_corpus_path,
         '--test_documents=' + test_corpus_path,
         '--commons=' + commons_path],
        stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    print("Evaluation failed: ", e.returncode, e.output)
    return {'eval_metric': 0.0}

  eval_output = {}
  for line in output.splitlines():
    line = line.rstrip()
    print "Evaluation Metric: ", line
    parts = line.split('\t')
    assert len(parts) == 2, "%r" % line
    eval_output[parts[0]] = float(parts[1])
    if line.startswith("SLOT_F1"):
      eval_output['eval_metric'] = float(parts[1])

  assert eval_output.has_key('eval_metric'), "%r" % str(eval_output)
  return eval_output


def now():
  return "[" + str(datetime.now()) + "]"


def mem():
  p = psutil.Process(os.getpid())
  return str(p.memory_info())


def argparser(description):
  import argparse

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--output_folder', type=str)
  parser.add_argument('--commons', type=str)
  parser.add_argument('--train_corpus', type=str)
  parser.add_argument('--dev_corpus', type=str)
  parser.add_argument('--word_embeddings', type=str)

  # Training hyperparameters.
  # Notable omissions: decay_steps, dropout_rate.
  parser.add_argument('--train_steps', type=int)
  parser.add_argument('--report_every', type=int)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--learning_method', type=str)
  parser.add_argument('--use_moving_average', type=bool)
  parser.add_argument('--moving_average_coeff', type=float)
  parser.add_argument('--gradient_clip_norm', type=float)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--adam_beta1', type=float)
  parser.add_argument('--adam_beta2', type=float)
  parser.add_argument('--adam_eps', type=float)
  parser.add_argument('--l2_coeff', type=float)

  return parser


class Resources:
  def __init__(self):
    self.commons_path = None
    self.commons = None
    self.schema = None
    self.train = None
    self.spec = None


  def load(self,
           commons_path,
           train_path,
           word_embeddings_path=None):
    print "Loading training resources"
    print "Initial memory usage", mem()
    self.commons_path = commons_path
    self.commons = sling.Store()
    self.commons.load(commons_path)
    self.commons.freeze()
    self.schema = sling.DocumentSchema(self.commons)

    self.train = Corpora(
        train_path, self.commons, self.schema, gold=True, loop=False)
    print "Pointed to training corpus in", train_path, mem()

    self.spec = Spec()
    self.spec.commons_path = commons_path
    self.spec.build(self.commons, self.train)
    print "After building spec", mem()

    if word_embeddings_path is not None:
      self.spec.load_word_embeddings(word_embeddings_path)
      print "After loading pre-trained word embeddings", mem()

