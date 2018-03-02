// Copyright 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SLING_MYELIN_LEARNING_H_
#define SLING_MYELIN_LEARNING_H_

#include <string>

#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"

namespace sling {
namespace myelin {

// Cross entropy loss for multi-class classification.
class CrossEntropyLoss {
 public:
  CrossEntropyLoss(const string &name = "loss");

  // Build loss function together with gradient computation.
  void Build(Flow *flow, Flow::Variable *logits);

  // Initialize loss for model.
  void Initialize(const Network &network);

  // Batch for accumulating losses from forward computations.
  class Batch {
   public:
    // Initialize loss batch computation.
    Batch(const CrossEntropyLoss &loss);

    // Clear accumulated losses.
    void Clear();

    // Accumulate loss from logits.
    void Forward(float *logits, int target);

    // Compute loss gradient for batch.
    void Backward();

    // Return average loss for batch.
    float loss() { return *backward_.Get<float>(loss_.loss_); }

    // Return current batch size.
    int batch_size() { return *forward_.Get<int>(loss_.batch_size_); }

    // Return loss gradient.
    float *dlogits() { return backward_.Get<float>(loss_.dlogits_); }

   private:
    const CrossEntropyLoss &loss_;

    Instance forward_;   // forward computation and accumulation of losses
    Instance backward_;  // backward computation of gradient
  };

 private:
  // Name of loss function.
  string name_;

  // Name of gradient function for loss.
  string gradient_name_;

  // Cells for forward and backward loss computation.
  Cell *forward_ = nullptr;
  Cell *backward_ = nullptr;

  // Tensors for forward and backward loss computation.
  Tensor *logits_ = nullptr;
  Tensor *target_ = nullptr;
  Tensor *batch_size_ = nullptr;
  Tensor *primal_ = nullptr;
  Tensor *loss_ = nullptr;
  Tensor *dlogits_ = nullptr;
};

// A parameter optimizer applies updates to the learnable parameters of a model
// based on the (accumulated) gradients from backpropagation.
class Optimizer {
 public:
  virtual ~Optimizer() = default;

 private:
};

// Stocastic gradient descent optimizer.
class GradientDescentOptimizer : public Optimizer {
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_BUILDER_H_

