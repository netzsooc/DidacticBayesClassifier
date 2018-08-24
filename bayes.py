import pathlib
import itertools
from collections import Counter, defaultdict

def main():
    textos_spam = carga('textos_spam.txt')
    textos_ham = carga('textos_ham.txt')
    cantidad_spam, vocabulario_spam = obtener_vocab(textos_spam)
    cantidad_ham, vocabulario_ham = obtener_vocab(textos_ham)
    cantidad_total = cantidad_ham + cantidad_spam
    prior_spam = cantidad_spam / cantidad_total
    prior_ham = cantidad_ham / cantidad_total
    probs = {'ham':{'vocab':defaultdict(float, 
                            {w:(v/sum(vocabulario_ham.values())) 
                            for w, v in vocabulario_ham.items()}),
                    'prior': prior_ham},
             'spam':{'vocab':defaultdict(float,
                            {w:(v/sum(vocabulario_spam.values())) 
                            for w, v in vocabulario_spam.items()}),
                     'prior': prior_spam}}

def carga(archivo: pathlib.Path):
    with open(archivo) as f:
        for linea in f:
            yield linea.strip()


def obtener_vocab(documentos):
    vocab = Counter()
    for i, documento in enumerate(documentos):
        vocab += Counter(documento.split())
    return i + 1, vocab


def total_prob(evento: list, probs: dict):
    return sum(joint_prob(evento, categ) for categ in probs.values())


def joint_prob(evento: list, cond: dict):
    return cond['prior'] * prod([cond['vocab'][w] for w in evento])


def prod(elementos):
    if len(elementos) < 2:
        return elementos[0]
    else:
        return elementos[0] * prod(elementos[1:])


def bayes_theorem(a, b, probs):
    return joint_prob(b, a) / total_prob(b, probs)

if __name__ == '__main__':
    main()
