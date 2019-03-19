#!/usr/bin/env python3

from util import read_data, preprocess, process_test_data
from ngram import Ngram

if __name__ == '__main__':
    train_filename = '../data/brown.train.txt'
    validation_filename = '../data/brown.dev.txt'
    test_filename = '../data/brown.test.txt'
    train_data = read_data(train_filename)
    validation_data = read_data(validation_filename)
    test_data = read_data(test_filename)
    test_data = process_test_data(test_data)

    ngram = Ngram(3)
    print("TRAINING STARTED...")
    list_of_bigrams, unigram_counts, bigram_counts, list_of_trigrams, trigram_counts = ngram.train(train_data)
    one_gram_prob = ngram.calculate_onegram_prob(unigram_counts)
    bigram_prob = ngram.calculate_bigram_prob(list_of_bigrams, unigram_counts, bigram_counts)
    trigram_prob = ngram.calculate_trigram_prob(list_of_trigrams, bigram_counts, trigram_counts)
    one_gram_add_one_prob = ngram.onegram_add_one_smothing(unigram_counts)
    ngram.bigram_good_turing, ngram.bigram_zero_occurence_prob, ngram.bigram_good_turing_cstar = ngram.good_turing_smooting(list_of_bigrams, bigram_counts, len(list_of_bigrams),
                                                                                                                            '../output/good_turing_smooting_bigram.txt',
                                                                                                                            '../output/good_turing_smooting_bigram_result.txt')
    ngram.trigram_good_turing, ngram.trigram_zero_occurence_prob, ngram.trigram_good_turing_cstar = ngram.good_turing_smooting(list_of_trigrams, trigram_counts, len(list_of_trigrams),
                                                                                                                               '../output/good_turing_smooting_trigram.txt',
                                                                                                                               '../output/good_turing_smooting_trigram_result.txt')
    print("\n------UNIGRAM TEST DATA--------")
    ppl_score = ngram.ppl(test_data)
    ppl_score_with_smoohting = ngram.ppl_unigram_smoohted(test_data)
    bigram_ppl_score = ngram.ppl_bigram(test_data)
    trigram_ppl_score = ngram.ppl_trigram(test_data)
    print("perplexity score of test data:", ppl_score, ppl_score_with_smoohting, bigram_ppl_score, trigram_ppl_score)

    print("\n------INTERPOLATION--------")
    lambda_set = []
    lambda_set.append([0.5, 0.3, 0.2])
    lambda_set.append([0.8, 0.1, 0.1])
    lambda_set.append([0.1, 0.8, 0.1])
    lambda_set.append([0.1, 0.1, 0.8])
    lambda_set.append([0.6, 0.2, 0.2])
    lambda_set.append([0.2, 0.6, 0.2])
    lambda_set.append([0.2, 0.2, 0.6])
    lambda_set.append([0.4, 0.3, 0.3])
    lambda_set.append([0.3, 0.4, 0.3])
    lambda_set.append([0.3, 0.3, 0.4])
    lambda_set.append([0.2, 0.4, 0.4])
    lambda_set.append([0.4, 0.2, 0.4])
    lambda_set.append([0.4, 0.4, 0.2])
    lambda_set.append([0.1, 0.4, 0.5])
    lambda_set.append([0.1, 0.3, 0.6])
    lambda_set.append([0.1, 0.2, 0.7])
    lambda_set.append([0.05, 0.15, 0.8])
    lambda_set.append([0.05, 0.05, 0.9])
    for s in lambda_set:
        print("Lambda_Set: ", lambda_set)
        ppl_score = ngram.ppl_interpolation(test_data, s)
        print("perplexity score of test data:", ppl_score)
