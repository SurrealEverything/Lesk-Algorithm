import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk


def tokenize_gloss(gloss):

    tok_gloss = nltk.word_tokenize(gloss)
    tok_gloss = [
        tok.lower() for tok in tok_gloss if tok.isalpha()
    ]

    return tok_gloss


def lesk_measure(gloss1, gloss2):

    gloss1_set = set(gloss1)
    gloss2_set = set(gloss2)

    return len(gloss1_set.intersection(gloss2_set))


def original_lesk(sentence, word, pos=None, print_candidates=True):
    '''
    Word sense disambiguation using the original Lesk algorithm:

    @inproceedings{10.1145/318723.318728,
    author = {Lesk, Michael},
    title = {Automatic Sense Disambiguation Using Machine Readable Dictionaries: How to Tell a Pine Cone from an Ice Cream Cone},
    year = {1986},
    isbn = {0897912241},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/318723.318728},
    doi = {10.1145/318723.318728},
    booktitle = {Proceedings of the 5th Annual International Conference on Systems Documentation},
    pages = {24–26},
    numpages = {3},
    location = {Toronto, Ontario, Canada},
    series = {SIGDOC ’86}
    }
    '''

    sentence = [tok.lower() for tok in sentence if tok.isalpha()]
    best_sense = None
    max_overlap = 0

    if print_candidates:
        print('Candidates:\n')

    for sense in wordnet.synsets(word, pos=pos):

        sense_gloss = tokenize_gloss(sense.definition())
        overlap = 0

        for tok in sentence:

            for tok_sense in wordnet.synsets(tok):

                tok_sense_gloss = tokenize_gloss(tok_sense.definition())

                overlap += lesk_measure(sense_gloss, tok_sense_gloss)

        if print_candidates:
            print('def:', sense.definition())
            print('score:', overlap)
            print('\n')

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


def extended_lesk_measure(gloss1, gloss2):

    score = 0
    gloss1 = gloss1.copy()
    gloss2 = gloss2.copy()
    removing = True

    while removing:

        removing = False

        for subset_len in reversed(range(1, len(gloss1) + 1)):

            for left_offset in range(len(gloss1) + 1 - subset_len):

                left = gloss1[left_offset:left_offset + subset_len]

                for right_offset in range(len(gloss2) + 1 - subset_len):

                    right = gloss2[right_offset:right_offset + subset_len]

                    if left == right and '*' not in left and '*' not in right:

                        score += subset_len * subset_len
                        gloss1[left_offset:left_offset + subset_len] = ['*']
                        gloss2[right_offset:right_offset + subset_len] = ['*']
                        removing = True
                        break

    return score


def extended_list(synset):
    '''
    Returns a list with the glosses of synsets in a wordnet rel with the
    input synset, concatenated if a collision is found.
    '''
    extended_list = []

    synset_gloss = tokenize_gloss(synset.definition())

    extended_list.append(synset_gloss)

    rels = [
        'hypernyms', 'hyponyms', 'part_meronyms', 'substance_meronyms',
        'member_meronyms', 'part_holonyms', 'substance_holonyms',
        'member_holonyms', 'attributes', 'similar_tos', 'also_sees'
    ]

    for r in rels:
        r_synsets = getattr(synset, r)()
        r_gloss = ''

        for r_synset in r_synsets:
            r_gloss += r_synset.definition()
            r_gloss += ' '

        tok_r_gloss = tokenize_gloss(r_gloss)

        if tok_r_gloss:
            extended_list.append(tok_r_gloss)

    return extended_list


def extended_lesk(sentence, word, pos=None, print_candidates=True):
    '''
    Word sense disambiguation using the extended Lesk algorithm:

    @article{article,
    author = {Banerjee, Satanjeev and Pedersen, Ted},
    year = {2003},
    month = {05},
    pages = {},
    title = {Extended Gloss Overlaps as a Measure of Semantic Relatedness},
    journal = {IJCAI-2003}
    }
    '''

    sentence = [tok.lower() for tok in sentence if tok.isalpha()]
    best_sense = None
    max_overlap = 0

    if print_candidates:
        print('Candidates:\n')

    for sense in wordnet.synsets(word, pos=pos):

        overlap = 0
        extended_sense_gloss_lis = extended_list(sense)

        for sense_gloss in extended_sense_gloss_lis:

            for tok in sentence:

                for tok_sense in wordnet.synsets(tok):

                    extended_tok_sense_gloss_lis = extended_list(tok_sense)

                    for tok_sense_gloss in extended_tok_sense_gloss_lis:

                        overlap += extended_lesk_measure(
                            sense_gloss, tok_sense_gloss
                        )

        if print_candidates:
            print('def:', sense.definition())
            print('score:', overlap)
            print('\n')

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


sentence = nltk.word_tokenize(
    'Students enjoy going to school, studying and reading books'
)
word = 'school'

sense_simple = lesk(sentence, word, 'n')
print(sense_simple)
print(sense_simple.definition())
print(sense_simple.examples()[0], '\n')

sense_original = original_lesk(sentence, word, 'n')
print(sense_original)
print(sense_original.definition())
print(sense_original.examples()[0], '\n')

sense_extended = extended_lesk(sentence, word, 'n')
print(sense_extended)
print(sense_extended.definition())
print(sense_extended.examples()[0])
