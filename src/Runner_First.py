#!/usr/bin/env python3

from util import read_data, preprocess, process_test_data
from ngram import Ngram

if __name__ == '__main__':
    train_filename = '../data/train.txt'
    train_data = read_data(train_filename)

    ngram = Ngram(3)
    list_of_bigrams, unigram_counts, bigram_counts, list_of_trigrams, trigram_counts = ngram.train(train_data)
    one_gram_prob = ngram.calculate_onegram_prob(unigram_counts)
    bigram_prob = ngram.calculate_bigram_prob(list_of_bigrams, unigram_counts, bigram_counts)
    trigram_prob = ngram.calculate_trigram_prob(list_of_trigrams, bigram_counts, trigram_counts)
    one_gram_add_one_prob = ngram.onegram_add_one_smothing(unigram_counts)

    ngram.unigram_good_turing, ngram.unigram_zero_occurence_prob, ngram.unigram_good_turing_cstar = ngram.good_turing_smooting(list(unigram_counts.keys()), unigram_counts, len(list(unigram_counts.keys())),
                                                                                                                            '../output/good_turing_smooting_unigram.txt',
                                                                                                                            '../output/good_turing_smooting_unigram_result.txt')
    ngram.bigram_good_turing, ngram.bigram_zero_occurence_prob, ngram.bigram_good_turing_cstar = ngram.good_turing_smooting(list_of_bigrams, bigram_counts, len(list_of_bigrams),
                                                                                                                            '../output/good_turing_smooting_bigram.txt',
                                                                                                                            '../output/good_turing_smooting_bigram_result.txt')
    ngram.trigram_good_turing, ngram.trigram_zero_occurence_prob, ngram.trigram_good_turing_cstar = ngram.good_turing_smooting(list_of_trigrams, trigram_counts, len(list_of_trigrams),
                                                                                                                               '../output/good_turing_smooting_trigram.txt',
                                                                                                                               '../output/good_turing_smooting_trigram_result.txt')

    sentence = "I am"
    sentence = preprocess(sentence)
    print("\n------UNIGRAM TEST--------")
    ngram.N = 1
    print(sentence, ":", ngram.prob(sentence))

    print("\n------UNIGRAM ADD ONE SMOOTHING TEST--------")
    ngram.N = 1
    print(sentence, ":", ngram.sprob(sentence))

    print("\n------BIGRAM TEST--------")
    ngram.N = 2
    print(sentence, ngram.prob(sentence))

    print("\n------BIGRAM GOOD-TURING-SMOOTING TEST--------")
    ngram.N = 2
    print(sentence, ":", ngram.sprob(sentence))

    sentence = "I am Sam"
    sentence = preprocess(sentence)
    print("\n------TRIGRAM TEST--------")
    ngram.N = 3
    print(sentence, ":", ngram.prob(sentence))

    print("\n------TRIGRAM GOOD-TURING-SMOOTING TEST--------")
    ngram.N = 3
    print(sentence, ":", ngram.sprob(sentence))

    print("\n----------------------------------------------------------------------------------------")

    print("\n------GENERATE NEXT WORD, N=1--------")
    ngram.N = 1
    print(ngram.next(None, None))

    print("\n------GENERATE NEXT WORD, N=2--------")
    ngram.N = 2
    print("I", "(FOLLOWED BY=)", ngram.next(None, "I"))

    print("\n------GENERATE NEXT WORD, N=3--------")
    ngram.N = 3
    print("I", "am", "(FOLLOWED BY=)", ngram.next("I", "am"))

    print("\n----------------------------------------------------------------------------------------")

    print("\n------UNIGRAM RANDOMLY GENERATE SENTENCES TEST--------")
    unigram_sentences = ngram.unigram_generate_sentences(5)
    ngram.N = 1
    for sentence in unigram_sentences:
        joined_sentence = " ".join(sentence)
        score = ngram.prob(joined_sentence)
        smoothed_score = ngram.sprob(joined_sentence)
        print(joined_sentence, ":", score, smoothed_score)
    ppl_score = ngram.ppl(unigram_sentences)
    ppl_score_with_smoohting = ngram.ppl_unigram_smoohted(unigram_sentences)
    bigram_ppl_score = ngram.ppl_bigram(unigram_sentences)
    trigram_ppl_score = ngram.ppl_trigram(unigram_sentences)
    print("perplexity score of 5 sentences:", ppl_score, ppl_score_with_smoohting, bigram_ppl_score, trigram_ppl_score)

    print("\n------BIGRAM RANDOMLY GENERATE SENTENCES TEST--------")
    bigram_sentences = ngram.bigram_generate_sentences(5)
    ngram.N = 2
    for sentence in bigram_sentences:
        joined_sentence = " ".join(sentence)
        score = ngram.prob(joined_sentence)
        smoothed_score = ngram.sprob(joined_sentence)
        print(joined_sentence, ":", score, smoothed_score)
    ppl_score = ngram.ppl(bigram_sentences)
    ppl_score_with_smoohting = ngram.ppl_unigram_smoohted(bigram_sentences)
    bigram_ppl_score = ngram.ppl_bigram(bigram_sentences)
    trigram_ppl_score = ngram.ppl_trigram(bigram_sentences)
    print("perplexity score of 5 sentences:", ppl_score, ppl_score_with_smoohting, bigram_ppl_score, trigram_ppl_score)

    print("\n------TRIGRAM RANDOMLY GENERATE SENTENCES TEST--------")
    trigram_sentences = ngram.trigram_generate_sentences(5)
    ngram.N = 3
    for sentence in trigram_sentences:
        joined_sentence = " ".join(sentence)
        score = ngram.prob(joined_sentence)
        smoothed_score = ngram.sprob(joined_sentence)
        print(joined_sentence, ":", score, smoothed_score)
    ppl_score = ngram.ppl(trigram_sentences)
    ppl_score_with_smoohting = ngram.ppl_unigram_smoohted(trigram_sentences)
    bigram_ppl_score = ngram.ppl_bigram(trigram_sentences)
    trigram_ppl_score = ngram.ppl_trigram(trigram_sentences)
    print("perplexity score of 5 sentences:", ppl_score, ppl_score_with_smoohting, bigram_ppl_score, trigram_ppl_score)
