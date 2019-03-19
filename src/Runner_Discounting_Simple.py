#!/usr/bin/env python3

from util import read_data, preprocess, process_test_data
from ngram import Ngram

if __name__ == '__main__':
    train_filename = '../data/train.txt'
    test_filename = '../data/test.txt'
    train_data = read_data(train_filename)
    test_data = read_data(test_filename)

    ngram = Ngram(3)
    list_of_bigrams, unigram_counts, bigram_counts, list_of_trigrams, trigram_counts = ngram.train(train_data)
    one_gram_prob = ngram.calculate_onegram_prob(unigram_counts)
    bigram_prob = ngram.calculate_bigram_prob(list_of_bigrams, unigram_counts, bigram_counts)
    trigram_prob = ngram.calculate_trigram_prob(list_of_trigrams, bigram_counts, trigram_counts)
    one_gram_add_one_prob = ngram.onegram_add_one_smothing(unigram_counts)

    ngram.unigram_good_turing, ngram.unigram_zero_occurence_prob, ngram.unigram_good_turing_cstar = ngram.good_turing_smooting(list(unigram_counts.keys()), unigram_counts,
                                                                                                                               len(list(unigram_counts.keys())),
                                                                                                                               '../output/good_turing_smooting_unigram.txt',
                                                                                                                               '../output/good_turing_smooting_unigram_result.txt')
    ngram.bigram_good_turing, ngram.bigram_zero_occurence_prob, ngram.bigram_good_turing_cstar = ngram.good_turing_smooting(list_of_bigrams, bigram_counts, len(list_of_bigrams),
                                                                                                                            '../output/good_turing_smooting_bigram.txt',
                                                                                                                            '../output/good_turing_smooting_bigram_result.txt')
    ngram.trigram_good_turing, ngram.trigram_zero_occurence_prob, ngram.trigram_good_turing_cstar = ngram.good_turing_smooting(list_of_trigrams, trigram_counts, len(list_of_trigrams),
                                                                                                                               '../output/good_turing_smooting_trigram.txt',
                                                                                                                               '../output/good_turing_smooting_trigram_result.txt')

    β = 0.5
    ngram.calculate_bigram_discounting(list_of_bigrams, bigram_counts, unigram_counts, β)
    ngram.calculate_trigram_discounting(list_of_trigrams, trigram_counts, bigram_counts, β)
    test_list_of_bigrams, test_unigram_counts, test_bigram_counts, test_list_of_trigrams, test_trigram_counts = ngram.train(test_data)
    print("Perplexity using bigram discounted", ngram.ppl_bigram_discounted(process_test_data(test_data), test_bigram_counts,   len(test_list_of_bigrams)))
    print("Perplexity using trigrams discounted", ngram.ppl_trigram_discounted(process_test_data(test_data), test_trigram_counts,   len(test_list_of_trigrams)))
