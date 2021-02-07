#ifndef __CTCDECODE_H_
#define __CTCDECODE_H_

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace ctcdecode {

// forward declaration of private class
class PathTrie;

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */

class Scorer {
public:
  Scorer(float alpha,
         float beta,
         const std::string &lm_path,
         const std::vector<std::string> &vocabulary);
  ~Scorer();

  float get_log_cond_prob(const std::vector<std::string> &words);

  float get_sent_log_prob(const std::vector<std::string> &words);

  // return the max order
  size_t get_max_order() const { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() const { return dict_size_; }

  // retrun true if the language model is character based
  bool is_character_based() const { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta);

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix);

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels);

  // language model weight
  float alpha;
  // word insertion weight
  float beta;

  // pointer to the dictionary of FST
  void *dictionary;

protected:
  // necessary setup: load language model, set char map, fill FST's dictionary
  void setup(const std::string &lm_path,
             const std::vector<std::string> &vocab_list);

  // load language model from given path
  void load_lm(const std::string &lm_path);

  // fill dictionary for FST
  void fill_dictionary(bool add_space);

  // set char map
  void set_char_map(const std::vector<std::string> &char_list);

  float get_log_prob(const std::vector<std::string> &words);

  // translate the vector in index to string
  std::string vec2str(const std::vector<int> &input);

private:
  void *language_model_;
  bool is_character_based_;
  size_t max_order_;
  size_t dict_size_;

  int SPACE_ID_;
  std::vector<std::string> char_list_;
  std::unordered_map<std::string, int> char_map_;

  std::vector<std::string> vocabulary_;
};


/* Struct for the beam search output, containing the tokens based on the vocabulary indices, and the timesteps
 * for each token in the beam search output
 */
struct Output {
    std::vector<int> tokens, timesteps;
};

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary_size: The size of the vocabulary.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/

std::vector<std::pair<float, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<float>> &probs_seq,
    size_t vocabulary_size,
    size_t beam_size,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0,
    int log_input = 0,
    Scorer *ext_scorer = nullptr);

/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary_size: The size of the vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<float, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<float>>> &probs_split,
    size_t vocabulary_size,
    size_t beam_size,
    size_t num_processes,
    float cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    size_t blank_id = 0,
    int log_input = 0,
    Scorer *ext_scorer = nullptr);

} // namespace ctcdecode

#endif  // __CTCDECODE_H_
