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

import sling
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "sling/nlp/parser/trainer")

import train_util

Var = autograd.Variable
torch.manual_seed(1)

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
      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum'))
      self.add_module('lr_lstm_embedding_' + f.name, embedding)
      self.lr_embeddings.append(embedding)

      embedding = nn.EmbeddingBag(f.vocab_size, f.dim, mode='sum'))
      self.add_module('rl_lstm_embedding_' + f.name, embedding)
      self.rl_embeddings.append(embedding)

    # Two LSTMs.
    input_dim = spec.lstm_input_dim
    self.lr_lstm = nn.LSTM(input_dim, spec.lstm_hidden_dim, num_layers=1)
    self.rl_lstm = nn.LSTM(input_dim, spec.lstm_hidden_dim, num_layers=1)

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

    # Feedforward unit. This is not a single nn.Sequential model since it does
    # not allow accessing the hidden layer's activation.
    self.ff_layer = nn.Linear(spec.ff_input_dim, spec.ff_hidden_dim, bias=True)
    self.ff_relu = nn.ReLU()
    self.ff_softmax = nn.Sequential(
      nn.Linear(spec.ff_hidden_dim, spec.num_actions, bias=True),
      nn.LogSoftmax()
    )
    self.loss_fn = nn.NLLLoss()
    print "Modules:", self


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


  def _ff_output(
      self, lr_lstm_output, rl_lstm_output, ff_activations, state, debug=False):
    assert len(ff_activations) == state.steps
    ff_input_parts = []
    ff_input_parts_debug = []

    # Fixed features.
    for f, bag in zip(self.spec.ff_fixed_features, self.ff_fixed_embeddings):
      raw_features = self.spec.raw_ff_fixed_Features(f, state)

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
      indices = self.spec.translated_ff_link_features(f, state)
      assert len(indices) == f.num_links

      # Figure out where we need to pick the activations from.
      activations = ff_activations
      if f.name == "lr" or f.name == "frame_end_lr":
        activations = lr_lstm_output
      elif f.name == "rl" or f.name == "frame_end_rl":
        activations = rl_lstm_output

      for index in indices:
        if index is not None:
          ff_input_parts.append(transform(activations[index]))
        else:
          ff_input_parts.append(Var(torch.zeros(1, f.dim)))

      if debug:
        link_debug[1].extend(indices)
        ff_input_parts_debug.append(link_debug)

    ff_input = torch.cat(ff_input_parts, 1)
    ff_hidden = self.ff_relu(self.ff_layer(ff_input))
    ff_activations.append(ff_hidden)
    softmax_output = self.ff_softmax(ff_hidden)
    return softmax_output.view(self.spec.num_actions), ff_input_parts_debug


  def _lstm_outputs(self, document):
    raw_features = self.spec.raw_lstm_features(document)
    length = len(document.tokens)

    # Each of {lr,rl}_lstm_inputs should have shape (length, lstm_input_dim).
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


  def forward(self, document, train=False, debug=False):
    # Compute LSTM outputs.
    lr_out, rl_out, _ = self._lstm_outputs(document)

    # Run FF unit.
    state = ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []

    if train:
      loss = Var(torch.FloatTensor([1]).zero_())
      for gold in document.gold:
        gold_index = actions.indices.get(gold, None)
        assert gold_index is not None, "Unknown gold action %r" % gold

        ff_output, _ = self._ff_output(lr_out, rl_out, ff_activations, state)
        gold_var = Var(torch.LongTensor([gold_index]))
        loss += self.loss_fn(ff_output.view(1, -1), gold_var)

        assert state.is_allowed(gold_index), "Disallowed gold action: %r" % gold
        state.advance(gold)
      loss = loss / len(document.gold)
      return loss
    else:
      if len(document.tokens) == 0: return state

      topk = 50 if 50 < self.spec.num_actions else self.spec.num_actions
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


  def trace(self, document):
    length = len(document.tokens)
    lr_out, rl_out, lstm_features = self._lstm_outputs(document)

    assert len(self.spec.lstm_features) == len(lstm_features)
    for f in lstm_features:
      assert len(f.offsets) == length

    print length, "tokens in document"
    for index, t in enumerate(document.tokens):
      print "Token", index, "=", t.text
    print

    state = ParserState(document, self.spec)
    actions = self.spec.actions
    ff_activations = []
    steps = 0
    for gold in document.gold:
      print "State:", state
      gold_index = actions.indices.get(gold, None)
      assert gold_index is not None, "Unknown gold action %r" % gold

      if state.current < state.end:
        print "Token:", state.current, "=", document.tokens[state.current].text
        for feature_spec, values in zip(self.spec.lstm_features, lstm_features):
          # Recall that 'values' has indices at all sequence positions.
          # We need to get the slice of feature indices at the current token.
          start = values.offsets[state.current - state.begin]
          end = None
          if state.current < state.end - 1:
            end = values.offsets[state.current - state.begin + 1]

          current = values.indices[start:end]
          print "  LSTM feature:", feature_spec.name, ", indices=", current,
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


def learn(sempar, corpora, dev, illustrate=False):
  # Pick a reasonably long sample document.
  sample_doc = corpora.documents[0]
  for d in corpora.documents:
    if d.size() > 5:
      sample_doc = d
      break

  num_epochs = 100
  l2_coeff = 0.0001
  batch_size = 1
  report_every = 100
  optimizer = optim.Adam(sempar.parameters(), weight_decay=l2_coeff)

  current_batch_size = 0
  batch_loss = Var(torch.FloatTensor([0.0]))
  optimizer.zero_grad()

  corpora.shuffle()
  for epoch in xrange(num_epochs):
    if epoch % batch_size == 0:
      if current_batch_size > 0:
        batch_loss /= current_batch_size
        print "Epoch", epoch, "BatchLoss", batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

      del batch_loss
      current_batch_size = 0
      batch_loss = Var(torch.FloatTensor([0.0]))
      optimizer.zero_grad()

    loss = sempar.forward(corpora.documents[epoch % corpora.size()], train=True)
    batch_loss += loss
    current_batch_size += 1
    if batch_size > 1: print "  Epoch", epoch, "BatchItemLoss", loss.data[0]

  # Process the partial batch (if any) at the end.
  if current_batch_size > 0:
    batch_loss /= current_batch_size
    print "Epoch", epoch, "BatchLoss", batch_loss.data[0]
    batch_loss.backward()
    optimizer.step()

  # See how the sample document performs on the trained model.
  if illustrate:
    print "Sample Document:"
    for t in sample_doc.tokens:
      print "Token", t.text
    for g in sample_doc.gold:
      print "Gold", g
    state = sempar.forward(sample_doc, train=False)
    for a in state.actions:
      print "Predicted", a


def trial_run():
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  commons = sling.Store()
  commons.load(path + "commons.new")
  commons.freeze()
  schema = sling.DocumentSchema(commons)

  train = Corpora()
  train.read(path + "train.rec", commons, schema, max_count=5)
  print train.size(), "train documents read"

  spec = Spec()
  spec.build(commons, train)
  sempar = Sempar(spec)
  learn(sempar, train, illustrate=True)

trial_run()
