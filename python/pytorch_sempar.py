import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sling
import zipfile

Var = autograd.Variable
torch.manual_seed(1)


class Lexicon:
  def __init__(self, min_count = 1, normalize_digits=True, oov_item="<UNKNOWN>"):
    self.min_count = min_count
    self.counts = {}
    self.oov_item = oov_item
    self.oov_index = 0   # OOV is always at position 0
    self.normalize_digits = normalize_digits

    
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
    self.item_to_index = {}
    self.index_to_item = {}
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
    

class Token:
  def __init__(self, word=None, breaklevel=None):
    self.word = word
    self.breaklevel = breaklevel
    self.start = 0
    self.length = 0
    

class Mention:
  def __init__(self):
    self.begin = None
    self.length = None
    self.evokes = []
  
  
class Document:
  def __init__(self, commons, encoded):
    self.tokens = []
    self.mentions = []
    self.themes = []
    self.gold = []
    self.store = sling.Store(commons)
    self.object = self.store.parse(encoded, binary=True)
    tokens = self.object['/s/document/tokens']
    for t in tokens:
      token = Token()
      token.word = t['/s/token/text']
      token.start = t['/s/token/start']
      token.length = t['/s/token/length']
      token.breaklevel = t['/s/token/break']
      self.tokens.append(token)
      
    for mention in self.object('/s/document/mention'):      
      m = Mention()
      m.begin = mention['/s/phrase/begin']
      m.length = mention['/s/phrase/length']
      if m.length is None:
        m.length = 1
      m.evokes = [e for e in mention('/s/phrase/evokes')]
      self.mentions.append(m)
    
    self.mentions.sort(cmp=lambda x, y: x.begin < y.begin or x.begin == y.begin and x.length >= y.length)


class Corpora:
  def __init__(self):
    self.documents = []
    
    
  def size(self):    
    return len(self.documents)
  
  
  def add(self, doc):
    self.documents.append(doc)
            
            
  def subset(self, start, end):
    s = Corpora()
    s.documents = self.documents[start:end]
    return s


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
     return (self.type, self.length, self.source, self.target, self.role, self.label)
   
     
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
         

class Edge:
  def __init__(self, incoming=None, role=None, value=None):
    self.incoming = incoming
    self.role = role
    self.neighbor = value
    self.inverse = None
    self.used = False
   
   
class FrameInfo:
  def __init__(self, handle):
    self.handle = handle
    self.type = None
    self.edges = []
    self.mention = None
    self.output = False
   
   
class RoughAction:
  def __init__(self, t=None):
    self.action = Action(t)
    self.info = None
    self.other_info = None
 
  
class TransitionGenerator:
  def __init__(self, commons):
    self.commons = commons
    self._id = commons["id"]
    self._isa = commons["isa"]
    self._is = commons["is"]
    self.thing_type = commons["/s/thing"]
    assert self.thing_type.isglobal()

  
  def is_ref(self, handle):
    return type(handle) == sling.Frame
  
  
  def init_info(self, frame, frame_info, initialized):
    if frame in initialized:
      return
    initialized[frame] = True
 
    info = frame_info.get(frame, None)
    if info is None:
      info = FrameInfo(frame)
      frame_info[frame] = info

    pending = []
    for role, value in frame:
       if not self.is_ref(value) or role == self._id or (role == self._isa and value.islocal()):
         continue
       if role == self._isa and info.type is None:
         info.type = value
       else:
         edge = Edge(incoming=False, role=role, value=value)
         info.edges.append(edge)
         if value == frame:
           edge.inverse = len(info.edges) - 1
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
       info.type = self.thing_type
       
    for p in pending:
       self.init_info(p, frame_info, initialized)
       
  
  def _translate(self, attention, rough_action):
    action = Action(t=rough_action.action.type)
    if rough_action.action.length is not None:
      action.length = rough_action.action.length
    if rough_action.action.role is not None:
      action.role = rough_action.action.role
      
    if action.type == Action.EVOKE:
       action.label = rough_action.info.type
    elif action.type == Action.REFER:
      action.target = attention.index(rough_action.info.handle)
    elif action.type == Action.EMBED:
      action.label = rough_action.info.type
      action.target = attention.index(rough_action.other_info.handle)
    elif action.type == Action.ELABORATE:
      action.label = rough_action.info.type
      action.source = attention.index(rough_action.other_info.handle)
    elif action.type == Action.CONNECT:
      action.source = attention.index(rough_action.info.handle)
      action.target = attention.index(rough_action.other_info.handle)
    elif action.type == Action.ASSIGN:
      action.source = attention.index(rough_action.info.handle)
      action.label = rough_action.action.label
     
    return action
  
  
  def _update(self, attention, rough_action):
    t = rough_action.action.type
    if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
      attention.insert(0, rough_action.info.handle)
    elif t == Action.REFER or t == Action.ASSIGN or t == Action.CONNECT:
      attention.remove(rough_action.info.handle)
      attention.insert(0, rough_action.info.handle)
   
   
  def generate(self, doc):
    frame_info = {}
    initialized = {}
    mentions_with_frames = []
    for m in doc.mentions:
      if len(m.evokes) > 0:
        mentions_with_frames.append(m)
        for evoked in m.evokes:
          self.init_info(evoked, frame_info, initialized)
          frame_info[evoked].mention = m
     
    for theme in doc.themes:
      self.init_info(theme, frame_info, initialized)
    
    rough_actions = []
    start = 0
    evoked = {}
    for m in mentions_with_frames:
      for i in xrange(start, m.begin):
        rough_actions.append(RoughAction(Action.SHIFT))        
      start = m.begin
   
      for frame in m.evokes:        
        rough_action = RoughAction()
        rough_action.action.length = m.length
        rough_action.info = frame_info[frame]  
        if frame not in evoked:
          rough_action.action.type = Action.EVOKE
          evoked[frame] = True        
        else:
          rough_action.action.type = Action.REFER                        
        rough_actions.append(rough_action)
  
    for index in xrange(start, len(doc.tokens)):
      rough_actions.append(RoughAction(Action.SHIFT))
       
    rough_actions.append(RoughAction(Action.STOP))    
    rough_actions.reverse()
    actions = []
    attention = []
    while len(rough_actions) > 0:
      rough_action = rough_actions[-1]
      rough_actions.pop()
      actions.append(self._translate(attention, rough_action))
      self._update(attention, rough_action)
      t = rough_action.action.type
      if t == Action.EVOKE or t == Action.EMBED or t == Action.ELABORATE:
        rough_action.info.output = True
        for e in rough_action.info.edges:
          if e.used:
            continue
          nb_info = frame_info.get(e.neighbor, None)
          if nb_info is None or not nb_info.output:
            continue
          rough_actions.append(RoughAction(Action.CONNECT))
          rough_actions[-1].action.role = e.role
          rough_actions[-1].info = nb_info if e.incoming else rough_action.info          
          rough_actions[-1].other_info = rough_action.info if e.incoming else nb_info
          e.used = True
          e.inverse.used = True
           
        for e in rough_action.info.edges:
          if e.used or not e.incoming:
            continue
          nb_info = frame_info.get(e.neighbor, None)
          if nb_info is None or nb_info.output or nb_info.mention is not None:
             continue            
          rough_actions.append(RoughAction(Action.EMBED))
          rough_actions[-1].action.role = e.role
          rough_actions[-1].info = nb_info
          rough_actions[-1].other_info = rough_action.info
          e.used = True
          e.inverse.used = True
         
        for e in rough_action.info.edges:
          if e.used or e.incoming:
            continue
          nb_info = frame_info.get(e.neighbor, None)
          if nb_info is None or nb_info.output or nb_info.mention is not None:
            continue            
          rough_actions.append(RoughAction(Action.ELABORATE))
          rough_actions[-1].action.role = e.role
          rough_actions[-1].info = nb_info
          rough_actions[-1].other_info = rough_action.info
          e.used = True
          e.inverse.used = True
         
        for e in rough_action.info.edges:
          if e.used or e.neighbor.islocal() or e.incoming:
            continue
          rough_actions.append(RoughAction(Action.ASSIGN))
          rough_actions[-1].info = rough_action.info
          rough_actions[-1].action.role = e.role
          rough_actions[-1].action.label = e.neighbor
          e.used = True          
         
    return actions
         
         
class FeatureSpec:
  def __init__(self, name, dim, vocab=None, activation=None, num_links=None):
    self.name = name
    self.dim = dim
    self.vocab_size = vocab            # for fixed features
    self.activation_size = activation  # for link features
    self.num_links = num_links         # for link features


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

  


class Spec:
  def __init__(self):
    self.words_normalize_digits = True
    self.words_min_count = 1
    self.suffixes_normalize_digits = False
    self.suffixes_min_count = 1
    self.suffixes_max_length = 3
    self.actions_percentile = 99

    self.lstm_hidden_dim = 256
    self.ff_hidden_dim = 128
    self.words_dim = 32
    self.suffixes_dim = 16
    self.roles_dim = 16

    self.history_limit = 4
    self.frame_limit = 5


  def _build_action_table(self, corpora):
    self.actions = Actions()
    generator = TransitionGenerator(self.commons)
    for doc in corpora.documents:
      actions = generator.generate(doc)
      for action in actions:
        self.actions.add(action)
        doc.gold.append(action)
            
    self.actions.prune(self.actions_percentile)
    self.num_actions = self.actions.size()
    print self.num_actions, "unique gold actions before pruning"
    print len(self.actions.roles), "unique roles in action before pruning"

    allowed = self.num_actions - sum(self.actions.disallowed)
    print "num allowed actions after pruning", allowed
    #for action, disallowed in zip(self.actions.table, self.actions.disallowed):
    #  if disallowed: print "disallowed", action


  def get_suffixes(self, word):
    unicode_chars = list(word.decode("utf-8"))
    output = []
    for l in xrange(1, self.suffixes_max_length + 1):
      if len(unicode_chars) >= l:
        output.append("".join(unicode_chars[-l:]))
    return output


  def build(self, commons, corpora):
    # Prepare lexical dictionaries.
    self.commons = commons
    self.words = Lexicon(self.words_min_count, self.words_normalize_digits)
    self.suffix = Lexicon(self.suffixes_min_count, self.suffixes_normalize_digits)
    for doc in corpora.documents:
      for token in doc.tokens:
        word = token.word
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
    ff = []
    if num_roles > 0:
      fl = self.frame_limit
      ff.append(FeatureSpec("in_roles", dim=self.roles_dim, vocab=num_roles * fl))
      ff.append(FeatureSpec("out_roles", dim=self.roles_dim, vocab=num_roles * fl))
      ff.append(FeatureSpec("unlabeled_roles", dim=self.roles_dim, vocab=fl * fl))
      ff.append(FeatureSpec("labeled_roles", dim=self.roles_dim, vocab=num_roles * fl * fl))
    self.ff_fixed_features = ff

    ff = []
    ff.append(FeatureSpec("frame_creation", dim=64, activation=self.ff_hidden_dim, num_links=self.frame_limit))
    ff.append(FeatureSpec("frame_focus", dim=64, activation=self.ff_hidden_dim, num_links=self.frame_limit))
    ff.append(FeatureSpec("frame_end_lr", dim=64, activation=self.lstm_hidden_dim, num_links=self.frame_limit))
    ff.append(FeatureSpec("frame_end_rl", dim=64, activation=self.lstm_hidden_dim, num_links=self.frame_limit))
    ff.append(FeatureSpec("lr", dim=32, activation=self.lstm_hidden_dim, num_links=1))
    ff.append(FeatureSpec("rl", dim=32, activation=self.lstm_hidden_dim, num_links=1))
    ff.append(FeatureSpec("history", dim=64, activation=self.ff_hidden_dim, num_links=self.history_limit))
    self.ff_link_features = ff
    
    self.ff_input_dim = sum([f.dim for f in self.ff_fixed_features])
    self.ff_input_dim += sum([f.dim * f.num_links for f in self.ff_link_features])
    print "FF input dim", self.ff_input_dim
    assert self.ff_input_dim > 0
      
    # TODO: Write the following:
    # - Lexicons with their flags (e.g. normalize digits, suffix length)
    # - Action table.
    # - Feature specs.
    #
    # Does it make sense to write all this in one giant encoded SLING frame?


class Feature:
  def __init__(self):
    self.indices = []
    self.offsets = []
    
  
  def add(self, index):
    self.indices.append(index)
    
    
  def new_offset(self):
    assert len(self.offsets) == 0 or self.offsets[-1] < len(self.indices)
    self.offsets.append(len(self.indices))
    

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
    s = "C:" + str(self.current) + " in [" + str(self.begin) + ", " + str(self.end) + ")"
    s += " " + str(len(self.frames)) + " frames"
    for index, f in enumerate(self.attention):
      if index == 10: break
      s += "\n   - Attn " + str(index) + ":" + str(f.type) + " Creation:" + str(f.creation)
      s += ", Focus:" + str(f.focus) + ", #Edges:" + str(len(f.edges))
      s += " (" + str(len(f.spans)) + " spans)"
      if len(f.spans) > 0:
        for span in f.spans:
          s += " [" + str(span.start) + ", " + str(span.end) + ")"
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
        if role_id is not None: self.graph.append((i, role_id, self.index(value)))


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
      if action.type == Action.REFER and action.target >= self.attention_size(): return False
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
        if t == action.label and role == action.role and value is target: return False
      return True
    elif action.type == Action.ELABORATE:
      if action.source >= self.attention_size(): return False
      source = self.attention[action.source]
      for t, role, value in self.elaborate:
        if t == action.label and role == action.role and value is source: return False
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

  
  def add_to_attention(self, f):
    f.focus = self.steps
    self.attention.insert(0, f)


  def refocus_attention(self, index):
    f = self.attention[index]
    f.focus = self.steps
    if index > 0: self.attention.insert(0, self.attention.pop(index))


  def make_span(self, length):
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
      s = self.make_span(action.length)
      f = ParserState.Frame(action.label)
      f.start = self.current
      f.end = self.current + action.length
      f.creation = self.steps
      f.spans.append(s)
      s.evoked.append(f)
      self.frames.append(f)
      self.add_to_attention(f)
    elif action.type == Action.REFER:
      f = self.attention[action.target]
      f.focus = self.steps
      s = self.make_span(action.length)
      s.evoked.append(f)
      f.spans.append(s)
      self.refocus_attention(action.target)
    elif action.type == Action.CONNECT:
      f = self.attention[action.source]
      f.edges.append((action.role, self.attention[action.target]))
      f.focus = self.steps
      self.refocus_attention(action.source)
    elif action.type == Action.EMBED:
      target = self.attention[action.target]
      f = ParserState.Frame(action.label)
      f.creation = self.steps
      f.focus = self.steps
      f.edges.append((action.role, target))
      self.add_to_attention(f)
      self.embed.append((action.label, action.role, target))
    elif action.type == Action.ELABORATE:
      source = self.attention[action.source]
      f = ParserState.Frame(action.label)
      f.creation = self.steps
      f.focus = self.steps
      source.edges.append((action.role, f))
      self.add_to_attention(f)
      self.elaborate.append((action.label, action.role, source))
    elif action.type == Action.ASSIGN:
      source = self.attention[action.source]
      source.focus = self.steps
      source.edges.append((action.role, action.label))
      self.refocus_attention(action.source)
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


class Sempar(nn.Module):  
  def __init__(self, spec):
    super(Sempar, self).__init__()
    self.spec = spec
    
    # Note: PyTorch has BiLSTM support, that will use a single word embedding, a single
    # suffix embedding and so on. We can experiment with it, but for now we allocate
    # two separate LSTMs with their own embeddings.
    
    # LSTM Embeddings.
    self.lr_embeddings = []
    self.rl_embeddings = []
    for f in spec.lstm_features:
      self.lr_embeddings.append(nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum'))
      self.rl_embeddings.append(nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum'))
      self.add_module('lr_lstm_embedding_' + f.name, self.lr_embeddings[-1])
      self.add_module('rl_lstm_embedding_' + f.name, self.rl_embeddings[-1])
    
    # Two LSTMs.
    self.lr_lstm = nn.LSTM(spec.lstm_input_dim, spec.lstm_hidden_dim, num_layers=1)
    self.rl_lstm = nn.LSTM(spec.lstm_input_dim, spec.lstm_hidden_dim, num_layers=1)
    
    # FF Embeddings and network.
    self.ff_fixed_embeddings = []
    self.ff_link_transforms = []
    for f in spec.ff_fixed_features:
      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.ff_fixed_embeddings.append(embedding)
      self.add_module('ff_fixed_embedding_' + f.name, embedding)
      
    for f in spec.ff_link_features:
      transform = nn.Linear(f.activation_size, f.dim)
      self.ff_link_transforms.append(transform)
      self.add_module('ff_link_transform_' + f.name, transform)
         
    # Feedforward unit. This is not a single sequential model since we need to
    # access the hidden layer's activation, and nn.Sequential doesn't allow that.
    self.ff_layer = nn.Linear(spec.ff_input_dim, spec.ff_hidden_dim, bias=True)
    self.ff_relu = nn.ReLU()
    self.ff_softmax = nn.Sequential(
      nn.Linear(spec.ff_hidden_dim, spec.num_actions, bias=True),
      nn.LogSoftmax()
    )
    self.loss_fn = nn.NLLLoss()
    print "Modules:", self

    
  def _raw_lstm_features(self, document):
    output = []
    for f in self.spec.lstm_features:
      features = Feature()
      output.append(features)
      if f.name == "word":
        for token in document.tokens:
          features.new_offset()
          features.add(self.spec.words.index(token.word))
      elif f.name == "suffix":
        for token in document.tokens:
          features.new_offset()
          for s in self.spec.get_suffixes(token.word):
            features.add(self.spec.suffix.index(s))
      else:
        raise ValueError("LSTM feature '", f.name, "' not implemented")
    return output
    
  
  def _embedding_lookup(self, embedding_bags, features):
    assert len(embedding_bags) == len(features)
    values = []
    for feature, bag in zip(features, embedding_bags):
      indices = Var(torch.LongTensor(feature.indices))
      offsets = Var(torch.LongTensor(feature.offsets))
      values.append(bag(indices, offsets))
    return torch.cat(values, 1)
  
   
  def _init_hidden(self):
    num_layers = 1
    batch_size = 1
    return (Var(torch.randn(num_layers, batch_size, self.spec.lstm_hidden_dim)),
        Var(torch.randn(num_layers, batch_size, self.spec.lstm_hidden_dim)))
  

  def _get_ff_output(self, lr_lstm_output, rl_lstm_output, ff_activations, state):
    assert len(ff_activations) == state.steps
    ff_input_parts = []

    role_graph = state.role_graph()
    num_roles = len(self.spec.actions.roles)
    fl = self.spec.frame_limit
    for f, bag in zip(self.spec.ff_fixed_features, self.ff_fixed_embeddings):
      raw_features = []
      if f.name == "in_roles":
        for e in role_graph:
          if e[2] is not None and e[2] < fl and e[2] >= 0:
            raw_features.append(e[2] * num_roles + e[1])
      elif f.name == "out_roles":
        for e in role_graph:
          raw_features.append(e[0] * num_roles + e[1])
      elif f.name == "unlabeled_roles":
        for e in role_graph:
          if e[2] is not None and e[2] < fl and e[2] >= 0:
            raw_features.append(e[0] * fl + e[2])
      elif f.name == "labeled_roles":
        for e in role_graph:
          if e[2] is not None and e[2] < fl and e[2] >= 0:
            raw_features.append(e[0] * fl * num_roles + e[2] * num_roles + e[1])
      else:
        raise ValueError("FF feature '", f.name, "' not implemented")

      embedded_features = None
      if len(raw_features) == 0:
        embedded_features = Var(torch.zeros(1, f.dim))
      else:
        embedded_features = bag(
          Var(torch.LongTensor(raw_features)), Var(torch.LongTensor([0])))
      ff_input_parts.append(embedded_features)

    for f, transform in zip(self.spec.ff_link_features, self.ff_link_transforms):
      if f.name == "history":
        for i in xrange(f.num_links):
          if i < len(ff_activations):
            activation = transform(ff_activations[-1 - i])
          else:
            activation = Var(torch.zeros(1, f.dim))
          ff_input_parts.append(activation)
      elif f.name == "lr":
        if state.current < state.end:
          activation = transform(lr_lstm_output[state.current - state.begin])
        else:
          activation = Var(torch.zeros(1, f.dim))
        ff_input_parts.append(activation)
      elif f.name == "rl":
        if state.current < state.end:
          activation = transform(rl_lstm_output[-1 - (state.current - state.begin)])
        else:
          activation = Var(torch.zeros(1, f.dim))
        ff_input_parts.append(activation)
      elif f.name == "frame_end_lr":
        for i in xrange(f.num_links):
          end = state.frame_end_inclusive(i)
          if end != -1:
            activation = transform(lr_lstm_output[end - state.begin])
          else:
            activation = Var(torch.zeros(1, f.dim))
          ff_input_parts.append(activation)
      elif f.name == "frame_end_rl":
        for i in xrange(f.num_links):
          end = state.frame_end_inclusive(i)
          if end != -1:
            activation = transform(rl_lstm_output[-1 - (end - state.begin)])
          else:
            activation = Var(torch.zeros(1, f.dim))
          ff_input_parts.append(activation)
      elif f.name == "frame_creation" or f.name == "frame_focus":
        for i in xrange(f.num_links):
          step = state.creation_step(i) if f.name == "frame_creation" else state.focus_step(i)
          if step != -1:
            activation = transform(ff_activations[step])
          else:
            activation = Var(torch.zeros(1, f.dim))
          ff_input_parts.append(activation)
        
    ff_input = torch.cat(ff_input_parts, 1)
    ff_hidden = self.ff_relu(self.ff_layer(ff_input))
    ff_activations.append(ff_hidden)
    return self.ff_softmax(ff_hidden).view(self.spec.num_actions)


  def _get_lstm_outputs(self, doc):
    raw_features = self._raw_lstm_features(doc)
    length = len(doc.tokens)

    # Each of {lr,rl}_lstm_inputs is length(doc) * lstm_input_dim.
    lr_lstm_inputs = self._embedding_lookup(self.lr_embeddings, raw_features)
    rl_lstm_inputs = self._embedding_lookup(self.rl_embeddings, raw_features)
    assert length == lr_lstm_inputs.size(0)
    assert length == rl_lstm_inputs.size(0)

    # LSTM expects an input of shape (sequence length, batch size, input dim).
    # In our case, batch size is 1.
    lr_input = lr_lstm_inputs.view(length, 1, -1)
    hidden = self._init_hidden()
    lr_out, _ = self.lr_lstm(lr_input, hidden)

    # Note: Negative strides are not supported, otherwise we would just do:
    #   rl_input = rl_lstm_inputs[::-1]
    inverse_indices = torch.arange(length - 1, -1, -1).long()
    rl_input = rl_lstm_inputs[inverse_indices].view(length, 1, -1)
    hidden = self._init_hidden()
    rl_out, _ = self.rl_lstm(rl_input, hidden)
  
    return (lr_out, rl_out, raw_features)


  def forward(self, doc, train=False):
    # Compute LSTM outputs.
    lr_out, rl_out, _ = self._get_lstm_outputs(doc)
  
    # Run FF unit.
    state = ParserState(doc, self.spec)
    actions = self.spec.actions
    ff_activations = []

    if train:
      loss = Var(torch.FloatTensor([1]).zero_())
      for gold_action in doc.gold:
        gold_index = actions.indices.get(gold_action, None)
        assert gold_index is not None, "Unknown gold action %r" % gold_action

        ff_output = self._get_ff_output(lr_out, rl_out, ff_activations, state)
        loss += self.loss_fn(ff_output.view(1, -1), Var(torch.LongTensor([gold_index])))

        assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold_action
        state.advance(gold_action)
      loss = loss / len(doc.gold)
      return loss
    else:
      if len(doc.tokens) == 0: return state
      index = actions.shift()
      k = 50 if 50 < self.spec.num_actions else self.spec.num_actions
      while index != actions.stop():
        ff_output = self._get_ff_output(lr_out, rl_out, ff_activations, state)

        # Find the highest scoring allowed action among the top-k.
        # If all top-k actions are disallowed, then use a fallback action.
        _, topk = torch.topk(ff_output, k)
        found = False
        rank = "(fallback)"
        for candidate in topk.view(-1).data:
          if not actions.disallowed[candidate] and state.is_allowed(candidate):
            rank = str(candidate)
            found = True
            index = candidate
            break
        if not found:
          # Fallback.
          index = actions.shift() if state.current < state.end else actions.stop()
        print "Predicted", actions.table[index], "at rank ", rank
        state.advance(actions.table[index])
      return state


  def trace(self, doc):
    length = len(doc.tokens)
    lr_out, rl_out, lstm_features = self._get_lstm_outputs(doc)

    assert len(self.spec.lstm_features) == len(lstm_features)
    for f in lstm_features:
      assert len(f.offsets) == length

    print length, "tokens in document"
    for index, t in enumerate(doc.tokens):
      print "Token", index, "=", t.word
    print

    state = ParserState(doc, self.spec)
    actions = self.spec.actions
    ff_activations = []
    steps = 0
    for gold_action in doc.gold:
      print "State:", state
      gold_index = actions.indices.get(gold_action, None)
      assert gold_index is not None, "Unknown gold action %r" % gold_action

      if state.current < state.end:
        print "Now at token", state.current, "=", doc.tokens[state.current].word
        for spec, f in zip(self.spec.lstm_features, lstm_features):
          start = f.offsets[state.current - state.begin]
          end = None
          if state.current < state.end - 1:
            end = f.offsets[state.current - state.begin + 1]
          print "  LSTM feature:", spec.name, ", indices=", f.indices[start:end]

      ff_output = self._get_ff_output(lr_out, rl_out, ff_activations, state)
      assert ff_output.view(1, -1).size(1) == self.spec.num_actions

      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold_action
      state.advance(gold_action)
      print "Step", steps, ": advancing using gold action", gold_action
      print
      steps += 1


def train(sempar, corpora):
  optimizer = optim.Adam(sempar.parameters())
  for epoch in xrange(2000):
    optimizer.zero_grad()
    loss = sempar.forward(corpora.documents[epoch % train.size()], train=True)
    print "Epoch", epoch, "Loss", loss.data[0]
    loss.backward()
    optimizer.step()

  doc = corpora.documents[0]
  for t in doc.tokens:
    print "Token", t.word
  for g in doc.gold:
    print "Gold", g
  state = sempar.forward(doc, train=False)
  for a in state.actions:
    print "Predicted", a


def trial_run():
  commons = sling.Store()
  commons.load("/home/grahul/sempar_ontonotes/commons.new")
  commons.freeze()
  train = Corpora()
  with zipfile.ZipFile("/home/grahul/sempar_ontonotes/train.zip", "r") as trainzip:
    for index, fname in enumerate(trainzip.namelist()):
      doc = Document(commons, trainzip.read(fname))
      if index <= 1000:
        train.add(doc)
        if index == 1000: break      
  print train.size(), "train docs read"
  spec = Spec()
  spec.build(commons, train)
  sempar = Sempar(spec)
  sempar.trace(train.documents[1])

trial_run()
