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

import math
import os
import random
import sling
import sys
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, "sling/nlp/parser/trainer")

import train_util as training
from functools import partial

from train_util import mem as mem
from train_util import now as now

sys.path.insert(0, "sling/nlp/parser/trainer/flow")
import nn as flownn
import flow as flow
import builder as builder

Param = nn.Parameter
Var = autograd.Variable

torch.manual_seed(1)
random.seed(0x31337)

global_debug=False

def fstr(var):
  if type(var) is tuple and len(var) == 1: var = var[0]

  dim = var.dim()
  if dim == 1 or (dim == 2 and (var.size(0) == 1 or var.size(1) == 1)):
    var = var.view(1, -1)
  ls = var.data.numpy().tolist()
  if type(ls[0]) is list: ls = ls[0]
  ls = ["%.9f" % x for x in ls]
  return "[" + ",".join(ls) + "]"


def dprint(prefix, *args):
  def isvar(a):
    return isinstance(a, Var) or \
        type(a) is tuple and len(a) == 1 and type(a[0]) is Var

  if global_debug:
    ls = [fstr(a) if isvar(a) else repr(a) for a in args]
    print "Debug=" + prefix + "=" + " ".join(ls)


def dsprint(prefix, arg, *args):
  dprint(prefix + "=" + str(arg), args)


class Projection(nn.Module):
  def __init__(self, num_in, num_out, bias=True):
    super(Projection, self).__init__()
    self.weight = Param(torch.randn(num_in, num_out))
    self.bias = None
    if bias:
      self.bias = Param(torch.randn(1, num_out))


  def init(self, weight_stddev, bias_const=None):
    self.weight.data.normal_()
    self.weight.data.mul_(weight_stddev)
    if bias_const is not None and self.bias is not None:
      self.bias.data.fill_(bias_const)


  def forward(self, x):
    if x.size()[0] != 1: x = x.view(1, -1)
    out = torch.mm(x, self.weight)
    if self.bias is not None:
      out = out + self.bias
    return out


  def __repr__(self):
    s = self.weight.size()
    return self.__class__.__name__ + "(in=" + str(s[0]) + \
        ", out=" + str(s[1]) + ", bias=" + str(self.bias is not None) + ")"


class LinkTransform(Projection):
  def __init__(self, activation_size, dim):
    super(LinkTransform, self).__init__(activation_size + 1, dim, bias=False)


  def forward(self, activation=None):
    if activation is None:
      return self.weight[-1].view(1, -1)  # last row
    else:
      return torch.mm(activation.view(1, -1), self.weight[0:-1])


  def __repr__(self):
    s = self.weight.size()
    return self.__class__.__name__ + "(input_activation=" + str(s[0] - 1) + \
        ", dim=" + str(s[1]) + ", oov_vector=" + str(s[1])+ ")"


class DragnnLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(DragnnLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self._x2i = Param(torch.randn(input_dim, hidden_dim))
    self._h2i = Param(torch.randn(hidden_dim, hidden_dim))
    self._c2i = Param(torch.randn(hidden_dim, hidden_dim))
    self._bi = Param(torch.randn(1, hidden_dim))

    self._x2o = Param(torch.randn(input_dim, hidden_dim))
    self._h2o = Param(torch.randn(hidden_dim, hidden_dim))
    self._c2o = Param(torch.randn(hidden_dim, hidden_dim))
    self._bo = Param(torch.randn(1, hidden_dim))

    self._x2c = Param(torch.randn(input_dim, hidden_dim))
    self._h2c = Param(torch.randn(hidden_dim, hidden_dim))
    self._bc = Param(torch.randn(1, hidden_dim))


  def forward_one_step(self, input_tensor, prev_h, prev_c):
    i_ait = torch.mm(input_tensor, self._x2i) + \
        torch.mm(prev_h, self._h2i) + \
        torch.mm(prev_c, self._c2i) + \
        self._bi
    i_it = torch.sigmoid(i_ait)
    i_ft = 1.0 - i_it
    i_awt = torch.mm(input_tensor, self._x2c) + \
        torch.mm(prev_h, self._h2c) + self._bc
    i_wt = torch.tanh(i_awt)
    ct = torch.mul(i_it, i_wt) + torch.mul(i_ft, prev_c)
    i_aot = torch.mm(input_tensor, self._x2o) + \
        torch.mm(ct, self._c2o) + torch.mm(prev_h, self._h2o) + self._bo
    i_ot = torch.sigmoid(i_aot)
    ph_t = torch.tanh(ct)
    ht = torch.mul(i_ot, ph_t)

    return (ht, ct)


  # input_tensors should be (Document Length x LSTM Input Dim).
  def forward(self, input_tensors):
    h = Var(torch.zeros(1, self.hidden_dim))
    c = Var(torch.zeros(1, self.hidden_dim))
    hidden = []
    cell = []

    length = input_tensors.size(0)
    for i in xrange(length):
      (h, c) = self.forward_one_step(input_tensors[i].view(1, -1), h, c)
      hidden.append(h)
      cell.append(c)

    return (hidden, cell)

  def __repr__(self):
    return self.__class__.__name__ + "(in=" + str(self.input_dim) + \
        ", hidden=" + str(self.hidden_dim) + ")"


class Sempar(nn.Module):
  def __init__(self, spec):
    super(Sempar, self).__init__()
    self.spec = spec

    # Note: PyTorch has BiLSTM support, that uses a single word embedding,
    # a single suffix embedding and so on. We can experiment with it, but for
    # now we allocate two separate LSTMs with their own embeddings.

    # LSTM Embeddings.
    self.lr_embeddings = []
    self.rl_embeddings = []
    for f in spec.lstm_features:
      lr_embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('lr_lstm_embedding_' + f.name, lr_embedding)
      self.lr_embeddings.append(lr_embedding)

      rl_embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.add_module('rl_lstm_embedding_' + f.name, rl_embedding)
      self.rl_embeddings.append(rl_embedding)

    # Two LSTM cells.
    self.lr_lstm = DragnnLSTM(spec.lstm_input_dim, spec.lstm_hidden_dim)
    self.rl_lstm = DragnnLSTM(spec.lstm_input_dim, spec.lstm_hidden_dim)

    # FF Embeddings and network.
    self.ff_fixed_embeddings = []
    self.ff_link_transforms = []
    for f in spec.ff_fixed_features:
      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum')
      self.ff_fixed_embeddings.append(embedding)
      self.add_module('ff_fixed_embedding_' + f.name, embedding)

    for f in spec.ff_link_features:
      transform = LinkTransform(f.activation_size, f.dim)
      self.ff_link_transforms.append(transform)
      self.add_module('ff_link_transform_' + f.name, transform)

    # Feedforward unit. This is not a single nn.Sequential model since it does
    # not allow accessing the hidden layer's activation.
    h = spec.ff_hidden_dim
    self.ff_layer = Projection(spec.ff_input_dim, spec.ff_hidden_dim)
    self.ff_relu = nn.ReLU()
    self.ff_softmax = Projection(spec.ff_hidden_dim, spec.num_actions)
    self.loss_fn = nn.CrossEntropyLoss()

    self.regularized_params = [self.ff_layer.weight]
    print "Modules:", self


  def initialize(self):
    for lr, rl, f in zip(
      self.lr_embeddings, self.rl_embeddings, self.spec.lstm_features):
      coeff = 1.0 / math.sqrt(f.dim)
      lr = lr.weight.data
      rl = rl.weight.data
      lr.normal_()
      lr.mul_(coeff)
      rl.normal_()
      rl.mul_(coeff)

      # Initialize with pre-trained word embeddings, if provided.
      if f.name == "words" and self.spec.word_embeddings is not None:
        indices = torch.LongTensor(self.spec.word_embedding_indices)
        data = torch.Tensor(self.spec.word_embeddings)
        data = F.normalize(data)  # normalize each row
        lr.index_copy_(0, indices, data)
        rl.index_copy_(0, indices, data)
        print "Overwrote", len(self.spec.word_embeddings), f.name, \
            "embedding vectors with normalized pre-trained vectors."

    for matrix, f in zip(self.ff_fixed_embeddings, self.spec.ff_fixed_features):
      matrix.weight.data.normal_()
      matrix.weight.data.mul_(1.0 / math.sqrt(f.dim))

    for t, f in zip(self.ff_link_transforms, self.spec.ff_link_features):
      t.init(1.0 / math.sqrt(f.dim))

    params = [self.ff_layer.weight, self.ff_softmax.weight]
    params += [p for p in self.lr_lstm.parameters()]
    params += [p for p in self.rl_lstm.parameters()]
    for p in params:
      p.data.normal_()
      p.data.mul_(1e-4)

    self.ff_layer.bias.data.fill_(0.2)
    self.ff_softmax.bias.data.fill_(0.0)


  def initialize_from_tf(self, tf_file):
    param_map = {}
    unseen = {}
    for name, p in self.named_parameters():
      p.data.zero_()
      param_map[name] = p
      unseen[p] = name
      print "Will try to initialize", name

    with open(tf_file, "r") as f:
      for line in f:
        parts = line.split("=")
        if len(parts) == 3 and parts[0] == "Init":
          param = parts[1]
          if param not in param_map:
            paramw = param + ".weight"
            if paramw not in param_map:
              print 'Ignoring unknown param', param
              continue
            else:
              param = paramw

          t = torch.Tensor(eval(parts[2]))
          if t.dim() == 1: t = t.view(1, -1)
          del unseen[param_map[param]]
          param_map[param].data = t
          print "Initialized", param, "with data of shape", param_map[param].data.size()

    if len(unseen) > 0:
      print "Didn't see values for:"
      for param, name in unseen.iteritems():
        print name
      assert False

    if global_debug:
      print "Final values:"
      for name, p in self.named_parameters():
        print name, "=", p.data

  def _embedding_lookup(self, embedding_bags, features):
    assert len(embedding_bags) == len(features)
    values = []
    for feature, bag in zip(features, embedding_bags):
      if not feature.has_multi and not feature.has_empty:
        # This case covers features that return exactly one value per call.
        # So this covers word features and all fallback features.
        indices = Var(torch.LongTensor(feature.indices))
        values.append(bag(indices.view(len(feature.indices), 1)))
      else:
        # Other features, e.g. suffixes, may return 0 or >1 ids.
        subvalues = []
        dim = bag.weight.size(1)
        for index, i in enumerate(feature.indices):
          if type(i) is int:  # one feature id
            subvalues.append(bag(Var(torch.LongTensor([i])).view(1, 1)))
          elif len(i) > 0:    # multiple feature ids
            subvalues.append(bag(Var(torch.LongTensor(i)).view(1, len(i))))
          else:               # no feature id
            subvalues.append(Var(torch.zeros(1, dim)))
        values.append(torch.cat(subvalues, 0))

    return torch.cat(values, 1)


  def _ff_output(
      self, lr_lstm_output, rl_lstm_output, ff_activations, state, debug=False):
    assert len(ff_activations) == state.steps
    ff_input_parts = []
    ff_input_parts_debug = []

    # Fixed features.
    for f, bag in zip(self.spec.ff_fixed_features, self.ff_fixed_embeddings):
      raw_features = self.spec.raw_ff_fixed_features(f, state)

      dprint("FF_Fixed_Ids_" + f.name, raw_features)
      embedded_features = None
      if len(raw_features) == 0:
        embedded_features = Var(torch.zeros(1, f.dim))
      else:
        embedded_features = bag(
          Var(torch.LongTensor(raw_features)), Var(torch.LongTensor([0])))
      ff_input_parts.append(embedded_features)
      if debug: ff_input_parts_debug.append((f, raw_features))

    # Link features.
    link_features = zip(self.spec.ff_link_features, self.ff_link_transforms)
    for f, transform in link_features:
      link_debug = (f, [])

      # Figure out where we need to pick the activations from.
      activations = ff_activations
      if f.name == "lr" or f.name == "frame_end_lr":
        activations = lr_lstm_output
      elif f.name == "rl" or f.name == "frame_end_rl":
        activations = rl_lstm_output

      # Get indices into the activations. Recall that missing indices are
      # indicated via None, and they map to the last row in 'transform'.
      indices = self.spec.translated_ff_link_features(f, state)
      assert len(indices) == f.num_links

      debug_indices = indices
      offset = 0
      if f.name == "frame_end_rl" or f.name == "rl":
        offset = state.document.size()
      elif f.name == "history":
        offset = state.steps

      before = []
      after = []
      debug_indices = []
      if global_debug:
        debug_indices = [-1 if i is None else i + offset for i in indices]
        dprint("FF_Link_Ids_" + f.name, debug_indices)
      for index in indices:
        activation = None
        if index is not None:
          assert index < len(activations), "%r" % index
          if index < 0: assert -index <= len(activations), "%r" % index
          activation = activations[index]
          if global_debug: before.append(activation.view(1, -1))
        else:
          if global_debug:
            before.append(Var(torch.Tensor([0] * f.activation_size)).view(1, -1))
        vec = transform.forward(activation)
        if global_debug: after.append(vec)
        ff_input_parts.append(vec)

      if global_debug:
        dprint("FF_Link_VecBefore_" + f.name, torch.cat(before, 1))
        dprint("FF_Link_VecTransformed_" + f.name, torch.cat(after, 1))
      if debug:
        link_debug[1].extend(indices)
        ff_input_parts_debug.append(link_debug)

    ff_input = torch.cat(ff_input_parts, 1).view(-1, 1)
    dprint("FF_Input", ff_input)
    ff_hidden = self.ff_layer(ff_input)
    ff_hidden = self.ff_relu(ff_hidden)
    dprint("FF_Hidden", ff_hidden)
    ff_activations.append(ff_hidden)
    softmax_output = torch.mm(
        ff_hidden, self.ff_softmax.weight) + self.ff_softmax.bias

    dprint("FF_Logits", softmax_output)
    return softmax_output.view(self.spec.num_actions), ff_input_parts_debug


  def _lstm_outputs(self, document):
    raw_features = self.spec.raw_lstm_features(document)
    length = document.size()

    if global_debug:
      for fspec, fvalues in zip(self.spec.lstm_features, raw_features):
        assert len(fvalues.indices) == length, len(fvalues.indices)
        for indices in fvalues.indices:
          dprint("LR_Fixed_Ids_" + fspec.name, indices)
        for indices in reversed(fvalues.indices):
          dprint("RL_Fixed_Ids_" + fspec.name, indices)

    # Each of {lr,rl}_inputs should have shape (length, lstm_input_dim).
    lr_inputs = self._embedding_lookup(self.lr_embeddings, raw_features)
    rl_inputs = self._embedding_lookup(self.rl_embeddings, raw_features)
    assert length == lr_inputs.size(0)
    assert length == rl_inputs.size(0)

    lr_out, _ = self.lr_lstm.forward(lr_inputs)

    # Note: Negative strides are not supported, otherwise we would just do:
    #   rl_input = rl_inputs[::-1]
    inverse_indices = torch.arange(length - 1, -1, -1).long()
    rl_inputs = rl_inputs[inverse_indices]
    rl_out, _ = self.rl_lstm.forward(rl_inputs)

    if global_debug:
      for i in xrange(length):
        dsprint("LR_Input", i, lr_inputs[i, :])
        dsprint("RL_Input", i, rl_inputs[i, :])
        dsprint("LR_Output", i, lr_out[i])
        dsprint("RL_Output", i, rl_out[i])

    return (lr_out, rl_out, raw_features)


  def forward(self, document, train=False, debug=False):
    # Compute LSTM outputs.
    lr_out, rl_out, _ = self._lstm_outputs(document)

    # Run FF unit.
    state = training.ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []

    if train:
      loss = Var(torch.FloatTensor([1]).zero_())
      for index, gold in enumerate(document.gold):
        gold_index = actions.indices.get(gold, None)
        assert gold_index is not None, "Unknown gold action %r" % gold

        ff_output, _ = self._ff_output(lr_out, rl_out, ff_activations, state)
        gold_var = Var(torch.LongTensor([gold_index]))
        step_loss = self.loss_fn(ff_output.view(1, -1), gold_var)
        dsprint("Stepcost", index, step_loss)
        loss += step_loss

        dsprint("Oracle", index, gold)
        assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
        state.advance(gold)
      return loss, len(document.gold)
    else:
      if document.size() == 0: return state

      topk = 7000
      if topk > self.spec.num_actions:
        topk = self.spec.num_actions
      shift = actions.shift()
      stop = actions.stop()
      predicted = shift
      while predicted != stop:
        ff_output, _ = self._ff_output(lr_out, rl_out, ff_activations, state)

        # Find the highest scoring allowed action among the top-k.
        # If all top-k actions are disallowed, then use a fallback action.
        _, topk_indices = torch.topk(ff_output, topk)
        found = False
        rank = "(fallback)"
        for candidate in topk_indices.view(-1).data:
          if not actions.disallowed[candidate] and state.is_allowed(candidate):
            rank = str(candidate)
            found = True
            predicted = candidate
            break
        if not found:
          # Fallback.
          predicted = shift if state.current < state.end else stop

        action = actions.table[predicted]
        state.advance(action)
        if debug:
          print "Predicted", action, "at rank ", rank

      return state


  def model_trace(self, document):
    length = document.size()
    lr_out, rl_out, lstm_features = self._lstm_outputs(document)

    assert len(self.spec.lstm_features) == len(lstm_features)
    for f in lstm_features:
      assert len(f.offsets) == length

    print length, "tokens in document"
    for index, t in enumerate(document.tokens()):
      print "Token", index, "=", t.text
    print

    state = training.ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []
    steps = 0
    for gold in document.gold:
      print "State:", state
      gold_index = actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action %r" % gold

      if state.current < state.end:
        print "Token", state.current, "=", document.tokens()[state.current].text
        for feature_spec, values in zip(self.spec.lstm_features, lstm_features):
          # Recall that 'values' has indices at all sequence positions.
          # We need to get the slice of feature indices at the current token.
          start = values.offsets[state.current - state.begin]
          end = None
          if state.current < state.end - 1:
            end = values.offsets[state.current - state.begin + 1]

          current = values.indices[start:end]
          print "  LSTM feature:", feature_spec.name, ", indices=", current,\
              "=", self.spec.lstm_feature_strings(current)

      ff_output, ff_debug = self._ff_output(
          lr_out, rl_out, ff_activations, state, debug=True)
      for f, indices in ff_debug:
        debug = self.spec.ff_fixed_features_debug(f, indices)
        print "  FF Feature", f.name, "=", str(indices), debug
      assert ff_output.view(1, -1).size(1) == self.spec.num_actions

      assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
      state.advance(gold)
      print "Step", steps, ": advancing using gold action", gold
      print
      steps += 1


  def dump_flow(self, flow_file):
    fl = flow.Flow()
    spec = self.spec
    spec.dump_flow(fl)

    def index_vars(bldr, feature_spec):
      return bldr.var(name=feature.name, dtype="int32", shape=[1, feature.num])


    def dump_fixed_feature(feature, bag, bldr, concat_op):
      indices = index_vars(bldr, feature)
      s = bag.weight.size()
      embedding = bldr.var(name=feature.name + "_embedding", shape=[s[0], s[1]])
      embedding.data = bag.weight.data.numpy()

      lookup = bldr.rawop(optype="Lookup", name=feature.name + "/Lookup")
      lookup.add_input(indices)
      lookup.add_input(embedding)
      embedded = bldr.var(name=feature.name + "_embedded", shape=[1, s[1]])
      lookup.add_output(embedded)
      concat_op.add_input(embedded)

    def finish_concat_op(bldr, op):
      op.add_attr("N", len(op.inputs))
      axis = bldr.const(1, "int32")
      op.add_input(axis)

    # Specify LSTMs and their input features.
    lr = builder.Builder(fl, "lr_lstm")
    rl = builder.Builder(fl, "rl_lstm")
    lr_input = lr.var(name="input", shape=[1, spec.lstm_input_dim])
    rl_input = rl.var(name="input", shape=[1, spec.lstm_input_dim])
    lr_out, lr_cnx = flownn.lstm(lr, input=lr_input, size=spec.lstm_hidden_dim)
    rl_out, rl_cnx = flownn.lstm(rl, input=rl_input, size=spec.lstm_hidden_dim)

    lr_concat_op = lr.rawop(optype="ConcatV2", name="concat")
    lr_concat_op.add_output(lr_input)
    rl_concat_op = rl.rawop(optype="ConcatV2", name="concat")
    rl_concat_op.add_output(rl_input)

    for i, feature in enumerate(spec.lstm_features):
      dump_fixed_feature(feature, self.lr_embeddings[i], lr, lr_concat_op)
      dump_fixed_feature(feature, self.rl_embeddings[i], rl, rl_concat_op)

    finish_concat_op(lr, lr_concat_op)
    finish_concat_op(rl, rl_concat_op)

    # Specify the FF unit.
    ff = builder.Builder(fl, "ff")
    ff_input = ff.var(name="input", shape=[1, spec.ff_input_dim])
    ff_logits, ff_hidden = flownn.feed_forward(
        ff, \
        input=ff_input, \
        layers=[spec.ff_hidden_dim, spec.num_actions], \
        hidden=0)
    ff_concat_op = ff.rawop(optype="ConcatV2", name="concat")
    ff_concat_op.add_output(ff_input)

    def link(bldr, name, dim, cnx):
      l = bldr.var("link/" + name, shape=[-1, dim])
      l.ref = True
      cnx.add(l)
      return l

    # Add links to the two LSTMs.
    ff_lr = link(ff, "lr_lstm", spec.lstm_hidden_dim, lr_cnx)
    ff_rl = link(ff, "rl_lstm", spec.lstm_hidden_dim, rl_cnx)
    print ff_lr, ff_rl, lr_cnx, rl_cnx

    # Add link and connector for previous steps.
    ff_cnx = ff.cnx("step", args=[])
    ff_steps = link(ff, "steps", spec.ff_hidden_dim, ff_cnx)

    for feature, bag in zip(spec.ff_fixed_features, self.ff_fixed_embeddings):
      dump_fixed_feature(feature, bag, ff, ff_concat_op)

    for feature, lt in zip(spec.ff_link_features, self.ff_link_transforms):
      indices = index_vars(ff, feature)

      activations = None
      n = feature.name
      if n == "frame_end_lr" or n == "lr":
        activations = ff_lr
      elif n == "frame_end_rl" or n == "rl":
        activations = ff_rl
      elif n in ["frame_creation_steps", "frame_focus_steps", "history"]:
        activations = ff_steps
      else:
        raise ValueError("Unknown feature %r" % n)

      name = feature.name + "/collect"
      collect = ff.rawop(optype="Collect", name=name)
      collect.add_input(indices)
      collect.add_input(activations)
      collected = ff.var(
          name=name + ":0", shape=[feature.num, activations.shape[1] + 1])
      collect.add_output(collected)

      sz = lt.weight.size()
      print n, sz
      transform = ff.var(name=feature.name + "/transform", shape=[sz[0], sz[1]])
      transform.data = lt.weight.data.numpy()

      name = feature.name + "/MatMul"
      matmul = ff.rawop(optype="MatMul", name=name)
      matmul.add_input(collected)
      matmul.add_input(transform)
      output = ff.var(name + ":0", shape=[feature.num, sz[1]])
      matmul.add_output(output)
      ff_concat_op.add_input(output)

    finish_concat_op(ff, ff_concat_op)
    fl.save(flow_file)
    print "Wrote flow to", flow_file


def dev_accuracy(commons_path, commons, dev_path, schema, tmp_folder, sempar):
  dev = training.Corpora(dev_path, commons, schema, gold=False, loop=False)
  print "Annotating dev documents", now(), mem()
  test_path = os.path.join(tmp_folder, "dev.annotated.rec")
  writer = sling.RecordWriter(test_path)
  count = 1
  start_time = time.time()
  for document in dev:
    state = sempar.forward(document, train=False)
    state.write()
    writer.write(str(count), state.encoded())
    if count % 100 == 0:
      print "  Annotated", count, "documents", now(), mem()
    count += 1
  writer.close()
  end_time = time.time()
  print "Annotated", count, "documents in", "%.1f" % (end_time - start_time), \
      "seconds", now(), mem()

  return training.frame_evaluation(gold_corpus_path=dev_path, \
                                   test_corpus_path=test_path, \
                                   commons_path=commons_path)


# A trainer reads one example at a time, till a count of num_examples is
# reached. For each example it computes the loss.
# After every 'batch_size' examples, it computes the gradient and applies
# it, with optional gradient clipping.
class Trainer:
  class Hyperparams:
    def __init__(self, lr=0.0005, num_examples=1000000, report_every=24000, \
                 l2_coeff=1e-4, batch_size=8, gradient_clip=1.0, \
                 optimizer="adam", adam_beta1=0.01, adam_beta2=0.999,
                 adam_eps=1e-5, moving_avg=True, moving_avg_coeff=0.9999):
      self.lr = lr
      self.num_examples = num_examples
      self.report_every = report_every
      self.l2_coeff = l2_coeff
      self.batch_size = batch_size
      self.gradient_clip = 1.0
      self.optimizer = optimizer
      self.adam_beta1 = adam_beta1
      self.adam_beta2 = adam_beta2
      self.adam_eps = adam_eps
      self.moving_avg = moving_avg
      self.moving_avg_coeff = moving_avg_coeff


  def __init__(self, sempar, evaluator=None, model_file=None, \
               hyperparams=Hyperparams()):
    self.model = sempar
    self.evaluator = evaluator
    self.hyperparams = hyperparams

    if hyperparams.optimizer == "sgd":
      self.optimizer = optim.SGD(
        sempar.parameters(),
        lr=self.hyperparams.lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False)
    elif hyperparams.optimizer == "adam":
      self.optimizer = optim.Adam(
          sempar.parameters(), lr=hyperparams.lr, weight_decay=0, \
              betas=(hyperparams.adam_beta1, hyperparams.adam_beta2), \
              eps=hyperparams.adam_eps)

    num_elements = 0
    for name, p in sempar.named_parameters():
      print name, "requires_grad", p.requires_grad, p.size()
      num_elements += torch.numel(p)
    print "num elements", num_elements

    self.current_batch_num_transitions = 0
    self.current_batch_size = 0
    self.batch_loss = Var(torch.FloatTensor([0.0]))
    self._reset()
    self.count = 0
    self.last_eval_count = 0

    self.checkpoint_metrics = []
    self.best_metric = None
    self.model_file = model_file

    self.averages = {}
    if hyperparams.moving_avg:
      for name, p in sempar.named_parameters():
        if p.requires_grad:
          self.averages[name] = p.data.clone()


  def _reset(self):
    self.current_batch_size = 0
    self.current_batch_num_transitions = 0
    del self.batch_loss
    self.batch_loss = Var(torch.FloatTensor([0.0]))
    self.optimizer.zero_grad()


  def process(self, example):
    loss, num_transitions = self.model.forward(example, train=True)
    self.current_batch_num_transitions += num_transitions
    self.batch_loss += loss
    self.current_batch_size += 1
    self.count += 1
    if self.current_batch_size == self.hyperparams.batch_size:
      self.update()
    if self.count % self.hyperparams.report_every == 0:
      self.evaluate()


  def clip_gradients(self):
    if self.hyperparams.gradient_clip is not None:
      for p in self.model.parameters():
        torch.nn.utils.clip_grad_norm([p], self.hyperparams.gradient_clip)


  def tf_name(self, name):
    if name == "ff_layer.weight": return "ff/weights_0:0"
    if name == "ff_layer.bias": return "ff/bias_0:0"
    if name == "ff_softmax.weight": return "ff/weights_softmax:0"
    if name == "ff_softmax.bias": return "ff/bias_softmax:0"

    name = name.replace("_lstm._", "_lstm/")
    name = name.replace("lstm_embedding", "lstm/fixed_embedding_matrix")
    name = name.replace("_fixed_embedding", "/fixed_embedding_matrix")
    name = name.replace("_link_transform", "/linked_embedding_matrix")
    if name.endswith(".weight"): name = name[0:-7]
    name = name + ":0"
    return name


  def print_norm(self):
    if global_debug:
      for name, p in self.model.named_parameters():
        dprint("Norm_", self.tf_name(name), torch.norm(p).data[0])


  def print_grad_norm(self):
    if global_debug:
      for name, p in self.model.named_parameters():
        if p.grad is not None:
          name = self.tf_name(name)
          dprint("GradNorm_" + name, torch.norm(p.grad).data[0])


  def update(self):
    if self.current_batch_size > 0:
      start = time.time()
      self.batch_loss /= self.current_batch_num_transitions

      l2 = Var(torch.Tensor([0.0]))
      if self.hyperparams.l2_coeff > 0.0:
        for p in self.model.regularized_params:
          l2 += 0.5 * self.hyperparams.l2_coeff * torch.sum(p * p)
        self.batch_loss += l2

      self.batch_loss /= 3.0  # for parity with TF
      value = self.batch_loss.data[0]
      self.batch_loss.backward()
      self.clip_gradients()
      self.optimizer.step()
      self.print_grad_norm()
      self._reset()
      end = time.time()
      num_batches = self.count / self.hyperparams.batch_size
      self.print_norm()

      if self.hyperparams.moving_avg:
        decay = self.hyperparams.moving_avg_coeff
        decay2 = (1.0 + num_batches) / (10.0 + num_batches)
        if decay > decay2: decay = decay2
        for name, p in self.model.named_parameters():
          if p.requires_grad and name in self.averages:
            diff = (self.averages[name] - p.data) * (1 - decay)
            self.averages[name].sub_(diff)

      print "BatchLoss after", "(%d" % num_batches, \
          "batches =", self.count, "examples):", value, \
          " incl. L2=", fstr(l2 / 3.0), \
          "(%.1f" % (end - start), "secs)", now(), mem()
      dsprint("BatchLoss", num_batches, value)


  def _swap_with_ema_parameters(self):
    if not self.hyperparams.moving_avg: return
    for name, p in self.model.named_parameters():
      if name in self.averages:
        tmp = self.averages[name]
        self.averages[name] = p.data
        p.data = tmp


  def evaluate(self):
    if self.evaluator is not None:
      if self.count != self.last_eval_count:
        self._swap_with_ema_parameters()

        metrics = self.evaluator(self.model)
        self.checkpoint_metrics.append((self.count, metrics))
        eval_metric = metrics["eval_metric"]
        print "Eval metric after", self.count, ":", eval_metric

        if self.model_file is not None:
          if self.best_metric is None or self.best_metric < eval_metric:
            self.best_metric = eval_metric
            best_file = self.model_file + ".best"
            torch.save(self.model.state_dict(), best_file)
            print "Updating best model at", best_file

        self.last_eval_count = self.count
        self._swap_with_ema_parameters()


def learn(sempar, corpora, evaluator=None, illustrate=False):
  # Pick a reasonably long sample document.
  sample_doc = None
  for d in corpora:
    if d.size() > 5:
      sample_doc = d
      break

  #if os.path.exists(model_file):
  #  sempar.load_state_dict(torch.load(model_file))
  #  print "Loaded model from", model_file

  model_file = "/tmp/pytorch.model"
  trainer = Trainer(sempar, evaluator, model_file)
  corpora.rewind()
  corpora.set_loop(True)
  for document in corpora:
    if trainer.count > trainer.hyperparams.num_examples:
      break
    trainer.process(document)

  # Process the partial batch (if any) at the end, and evaluate one last time.
  trainer.update()
  dprint("StartAnnotation")
  trainer.evaluate()
  dprint("EndAnnotation")

  # See how the sample document performs on the trained model.
  if illustrate:
    print "Sample Document:"
    for t in sample_doc.tokens():
      print "Token", t.text
    for g in sample_doc.gold:
      print "Gold", g
    state = sempar.forward(sample_doc, train=False)
    state.write()
    for a in state.actions:
      print "Predicted", a
    print state.textual()


def replicate():
  torch.set_printoptions(precision=12)
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  tf_folder = path + "/out-tf/"
  resources = training.Resources()
  resources.load(commons_path=path + "commons.new",
                 train_path=path + "dev5long.rec",
                 word_embeddings_path=path + "word2vec-32-embeddings.bin")
                 #word_embeddings_path=None) #path + "word2vec-32-embeddings.bin")

  sempar = Sempar(resources.spec)
  words = sempar.spec.words
  print "Words before override:", words.size()
  sempar.initialize_from_tf(tf_folder + "tf.debug")

  sempar.spec.words.load(tf_folder + "word-vocab")
  print "Words after override:", words.size()

  dev_path = path + "dev5.rec"
  tmp_folder = path + "pyt/tmp/"
  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      dev_path,
                      resources.schema,
                      tmp_folder)
  hyperparams = Trainer.Hyperparams(
      batch_size=1, num_examples=20, lr=0.05, l2_coeff=0.0, \
          gradient_clip=None, optimizer="sgd")
  trainer = Trainer(sempar, evaluator, None, hyperparams=hyperparams)

  resources.train.rewind()
  resources.train.set_loop(True)
  for document in resources.train:
    if trainer.count >= trainer.hyperparams.num_examples:
      break
    trainer.process(document)

  # Process the partial batch (if any) at the end, and evaluate one last time.
  trainer.update()
  global_debug=False
  trainer.evaluate()


def flow_test():
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  resources = training.Resources()
  resources.load(commons_path=path + "commons.new",
                 train_path=path + "dev500.rec",
                 word_embeddings_path=None)

  sempar = Sempar(resources.spec)
  sempar.initialize()

  flow_file = "/tmp/sempar.pyt.flow"
  sempar.dump_flow(flow_file)


def trial_run():
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  resources = training.Resources()
  resources.load(commons_path=path + "commons.new",
                 train_path=path + "train_shuffled.rec",
                 word_embeddings_path=path + "word2vec-32-embeddings.bin")

  sempar = Sempar(resources.spec)
  sempar.initialize()

  dev_path = path + "dev.rec"
  tmp_folder = path + "tmp/"
  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      dev_path,
                      resources.schema,
                      tmp_folder)
  learn(sempar, resources.train, evaluator, illustrate=False)


def affix_test():
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  resources = training.Resources()
  resources.load(commons_path=path + "commons.new",
                 train_path=path + "dev500.rec")
  print resources.spec.suffix.first_few("Affix ", n=1000)
  buf = resources.spec.dump_suffixes()
  with open("/tmp/affixes", "wb") as f:
    f.write(buf)
  resources.spec.actions.save(resources.commons, "/tmp/actions")


#affix_test()
#trial_run()
flow_test()
