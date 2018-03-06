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
import random
import sling
import sys
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, "sling/nlp/parser/trainer")

import train_util as training
from functools import partial

from train_util import mem as mem
from train_util import now as now

Var = autograd.Variable

torch.manual_seed(1)
random.seed(0x31337)

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

      # Initialize with pre-trained word embeddings, if provided.
      if f.name == "word" and spec.word_embeddings is not None:
        indices = torch.LongTensor(spec.word_embedding_indices)
        data = torch.Tensor(spec.word_embeddings)
        lr_embedding.weight.data.index_copy_(0, indices, data)
        rl_embedding.weight.data.index_copy_(0, indices, data)
        print "Overwrote", len(spec.word_embeddings), f.name, \
            "embedding vectors with pre-trained vectors."

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
      raw_features = self.spec.raw_ff_fixed_features(f, state)

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
          assert len(activations) > index, "%r" % index
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
    length = document.size()

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
    state = training.ParserState(document, self.spec)
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
  def __init__(self, sempar, evaluator=None, model_file=None):
    self.model = sempar
    self.evaluator = evaluator

    self.num_examples = 1200000
    self.report_every = 40000
    self.l2_coeff = 0.0001
    self.batch_size = 8
    self.gradient_clip = 1.0 # 'None' to disable clipping
    self.optimizer = optim.Adam(
      sempar.parameters(), lr=0.0005, weight_decay=self.l2_coeff, \
      betas=(0.01, 0.999), eps=1e-5)
    print "Trainer params", str(self.__dict__)
    print "Optimizer params", str(self.optimizer.__dict__)

    self.current_batch_size = 0
    self.batch_loss = Var(torch.FloatTensor([0.0]))
    self._reset()
    self.count = 0
    self.last_eval_count = 0

    self.checkpoint_metrics = []
    self.best_metric = None
    self.model_file = model_file


  def _reset(self):
    self.current_batch_size = 0
    del self.batch_loss
    self.batch_loss = Var(torch.FloatTensor([0.0]))
    self.optimizer.zero_grad()


  def process(self, example):
    loss = self.model.forward(example, train=True)
    self.batch_loss += loss
    self.current_batch_size += 1
    self.count += 1
    if self.current_batch_size == self.batch_size:
      self.update()
    if self.count % self.report_every == 0:
      self.evaluate()


  def update(self):
    if self.current_batch_size > 0:
      start = time.time()
      self.batch_loss /= self.current_batch_size
      self.batch_loss /= 3.0
      value = self.batch_loss.data[0]
      self.batch_loss.backward()
      if self.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm(
            self.model.parameters(), self.gradient_clip)
      self.optimizer.step()
      self._reset()
      end = time.time()
      print "BatchLoss after (", (self.count / self.batch_size), "batches, ", \
          self.count, "examples):", value, "(%.1f" % (end - start), "secs)", \
          now(), mem()


  def evaluate(self):
    if self.evaluator is not None:
      if self.num_examples == 0 or self.count != self.last_eval_count:
        metrics = self.evaluator(self.model)
        self.checkpoint_metrics.append((self.count, metrics))
        eval_metric = metrics["eval_metric"]
        print "Eval metric after", self.count, ":", eval_metric

        if self.count != self.last_eval_count:
          current_file = self.model_file + ".latest"
          torch.save(self.model.state_dict(), current_file)
          print "Saving latest model at", current_file
          if self.best_metric is None or self.best_metric < eval_metric:
            self.best_metric = eval_metric
            best_file = self.model_file + ".best"
            torch.save(self.model.state_dict(), best_file)
            print "Updating best model at", best_file
        self.last_eval_count = self.count



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
    if trainer.count > trainer.num_examples:
      break
    trainer.process(document)

  # Process the partial batch (if any) at the end, and evaluate one last time.
  trainer.update()
  trainer.evaluate()

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


def trial_run():
  path = "/usr/local/google/home/grahul/sempar_ontonotes/"
  resources = training.Resources()
  resources.load(commons_path=path + "commons.new",
                 train_path=path + "train_shuffled.rec",
                 word_embeddings_path=path + "word2vec-32-embeddings.bin")

  sempar = Sempar(resources.spec)

  dev_path = path + "dev.rec"
  tmp_folder = path + "tmp/"
  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      dev_path,
                      resources.schema,
                      tmp_folder)
  learn(sempar, resources.train, evaluator, illustrate=False)

trial_run()
