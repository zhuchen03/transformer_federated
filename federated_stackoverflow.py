# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated Stack Overflow next word prediction library using TFF."""

import functools
from typing import Callable, Optional

from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import keras_metrics
from utils import training_loop
from utils import training_utils
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models
from utils.datasets.tff_datasets_stackoverflow import load_data as tff_load_stackoverflow



def run_federated(
    model_type: str,
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    client_datasets_random_seed: Optional[int] = None,
    vocab_size: Optional[int] = 10000,
    num_oov_buckets: Optional[int] = 1,
    sequence_length: Optional[int] = 20,
    max_elements_per_user: Optional[int] = 1000,
    num_validation_examples: Optional[int] = 10000,
    embedding_size: Optional[int] = 96,
    latent_size: Optional[int] = 670,
    num_layers: Optional[int] = 1,
    dff: Optional[int] = 2048,
    shared_embedding: Optional[bool] = False,
    total_rounds: Optional[int] = 1500,
    experiment_name: Optional[str] = 'federated_so_nwp',
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    max_eval_batches: Optional[int] = None,
    **kwargs):
  """Runs an iterative process on the Stack Overflow next word prediction task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, a
      `client_weight_fn` and returns a `tff.templates.IterativeProcess`. The
      `model_fn` must return a `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    vocab_size: Integer dictating the number of most frequent words to use in
      the vocabulary.
    num_oov_buckets: The number of out-of-vocabulary buckets to use.
    sequence_length: The maximum number of words to take for each sequence.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset.
    num_validation_examples: The number of test examples to use for validation.
    embedding_size: The dimension of the word embedding layer.
    latent_size: The dimension of the latent units in the recurrent layers.
    num_layers: The number of stacked recurrent layers to use.
    shared_embedding: Boolean indicating whether to tie input and output
      embeddings.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    max_eval_batches: If set to a positive integer, evaluation datasets are
      capped to at most that many batches. If set to None or a nonpositive
      integer, the full evaluation datasets are used.
    **kwargs: Additional arguments configuring the training loop. For details
      on supported arguments, see
      `federated_research/utils/training_utils.py`.
  """
  if model_type == 'transformer':
    model_builder = functools.partial(
      stackoverflow_models.create_transformer_lm,
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      d_embed=embedding_size,
      d_model=latent_size,
      dff=dff,
      num_heads=8,
      num_layers=num_layers,
      max_position_encoding=10000,
      dropout=0.1,
      name='stackoverflow-transformer',
    )
  else:
    model_builder = functools.partial(
        stackoverflow_models.create_recurrent_model,
        vocab_size=vocab_size,
        num_oov_buckets=num_oov_buckets,
        embedding_size=embedding_size,
        latent_size=latent_size,
        num_layers=num_layers,
        shared_embedding=shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  def metrics_builder():
    return [
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_with_oov', masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
        # Notice BOS never appears in ground truth.
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov_or_eos',
            masked_tokens=[pad_token, eos_token] + oov_tokens),
        keras_metrics.NumBatchesCounter(),
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token])
    ]

  # train_clientdata, _, _ = tff.simulation.datasets.stackoverflow.load_data()
  train_clientdata, _, _ = tff_load_stackoverflow(cache_dir="datasets")

  # TODO(b/161914546): consider moving evaluation to use
  # `tff.learning.build_federated_evaluation` to get metrics over client
  # distributions, as well as the example weight means from this centralized
  # evaluation.
  _, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size=vocab_size,
      max_sequence_length=sequence_length,
      num_validation_examples=num_validation_examples,
      num_oov_buckets=num_oov_buckets)

  if max_eval_batches and max_eval_batches >= 1:
    validation_dataset = validation_dataset.take(max_eval_batches)
    test_dataset = test_dataset.take(max_eval_batches)

  train_dataset_preprocess_comp = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=stackoverflow_word_prediction.create_vocab(vocab_size),
      num_oov_buckets=num_oov_buckets,
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user)

  input_spec = train_dataset_preprocess_comp.type_signature.result.element

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  iterative_process = iterative_process_builder(
      tff_model_fn, client_weight_fn=client_weight_fn)

  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      train_dataset_preprocess_comp, iterative_process)

  training_process.get_model_weights = iterative_process.get_model_weights

  client_datasets_fn = training_utils.build_client_datasets_fn(
      dataset=train_clientdata,
      clients_per_round=clients_per_round,
      random_seed=client_datasets_random_seed)

  evaluate_fn = training_utils.build_centralized_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=validation_dataset,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  test_fn = training_utils.build_centralized_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=validation_dataset.concatenate(test_dataset),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
