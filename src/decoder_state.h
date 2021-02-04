#ifndef __CTCDECODE_DECODER_STATE_H_
#define __CTCDECODE_DECODER_STATE_H_

#include <string>
#include <utility>
#include <vector>

#include "ctcdecode.h"
#include "scorer.h"

class DecoderState
{
  int abs_time_step;
  int space_id;
  size_t beam_size;
  double cutoff_prob;
  size_t cutoff_top_n;
  size_t blank_id;
  int log_input;
  size_t vocabulary_size;
  Scorer *ext_scorer;

  std::vector<PathTrie*> prefixes;
  PathTrie root;

public:
  /* Initialize CTC beam search decoder for streaming
   *
   * Parameters:
   *     vocabulary_size: The size of the vocabulary.
   *     beam_size: The width of beam search.
   *     cutoff_prob: Cutoff probability for pruning.
   *     cutoff_top_n: Cutoff number for pruning.
   *     ext_scorer: External scorer to evaluate a prefix, which consists of
   *                 n-gram language model scoring and word insertion term.
   *                 Default null, decoding the input sample without scorer.
  */
  DecoderState(size_t vocabulary_size,
               size_t beam_size,
               double cutoff_prob,
               size_t cutoff_top_n,
               size_t blank_id,
               int log_input,
               Scorer *ext_scorer);
  ~DecoderState() = default;

  /* Process logits in decoder stream
   *
   * Parameters:
   *     probs: 2-D vector where each element is a vector of probabilities
   *               over alphabet of one time step.
  */
  void next(const std::vector<std::vector<double>> &probs_seq);

  /* Get current transcription from the decoder stream state
   *
   * Return:
   *     A vector where each element is a pair of score and decoding result,
   *     in descending order.
  */
  std::vector<std::pair<double, Output>> decode() const;
};

#endif  // __CTCDECODE_DECODER_STATE_H_
