"""Tests for text.py."""

import time

import pytest


TEST_CASES = [
    (
        '',
        []
    ),
    (
        'Ich kann zum Kauf nur raten',
        ['Ich kann zum Kauf nur raten']
    ),
    (
        'Ich kann zum Kauf nur raten.',
        ['Ich kann zum Kauf nur raten.']
    ),
    (
        'Gute Schnitt und gute Passform. Farbe kommt gut an.',
        ['Gute Schnitt und gute Passform.', 'Farbe kommt gut an.']
    ),
    (
        'Auch als Longtop gut zu tragen (ich bin nur 161cm) '
        'auf etwas engere Hosen/Jeans.',
        ['Auch als Longtop gut zu tragen (ich bin nur 161cm) '
         'auf etwas engere Hosen/Jeans.']
    ),
    (
        'Tja, aber ich brauch sie halt doch - auch kleine dicke '
        'Frauen mit "Bauch-Beine-Po-Problem" müssen was anziehen.',
        ['Tja, aber ich brauch sie halt doch - auch kleine dicke '
         'Frauen mit "Bauch-Beine-Po-Problem" müssen was anziehen.']
    ),
    (
        'Sehr empfehlenswert! Ich kann zum Kauf nur raten.',
        ['Sehr empfehlenswert!', 'Ich kann zum Kauf nur raten.'],
    ),
    (
        'es betont leider jede wölbun :(  Ansonsten wenn jemand',
        ['es betont leider jede wölbun :(', 'Ansonsten wenn jemand'],
    ),
    (
        'Gut kombinierbar mit z.B. gelb und weiß.',
        ['Gut kombinierbar mit z.B. gelb und weiß.']
    ),
    (
        'Material absolut o. k. Manchmal sind die Hosen zu lang.',
        ['Material absolut o. k.', 'Manchmal sind die Hosen zu lang.']
    ),
    (
        'da es ja 2 Stck. sind',
        ['da es ja 2 Stck. sind']
    ),
    (
        'Das war das 2. Mal diese Woche.',
        ['Das war das 2. Mal diese Woche.']
    ),
    (
        'Es paßt genau. Gr.32 fällt wie Gr.38 aus!',
        ['Es paßt genau.', 'Gr.32 fällt wie Gr.38 aus!'],
    ),
    (
        'das Material ist super,also super Qualität\ndie Größe Normal '
        'ist mir leider zu kurz',
        ['das Material ist super,also super Qualität',
         'die Größe Normal ist mir leider zu kurz']
    ),
    (
        'Farbe, Material usw. waren in Ordnung.',
        ['Farbe, Material usw. waren in Ordnung.']
    ),
    (
        'Tolles Material.Kommt auch gut bei Mitmenschen an.',
        ['Tolles Material.', 'Kommt auch gut bei Mitmenschen an.']
    ),
    (
        'Auch der Preiß stimmt!!!! nur zu empfehlen.',
        ['Auch der Preiß stimmt!', 'nur zu empfehlen.']
    ),
    (
        'Wichtig: Nicht nur Matraze austauschen.',
        ['Wichtig: Nicht nur Matraze austauschen.'],
    ),
    (
        'nicht so schön wie sonst...das Material is dünner und die '
        'Größe...als ob es fast ne Nr zu klein wäre',
        ['nicht so schön wie sonst.',
         'das Material is dünner und die Größe.',
         'als ob es fast ne Nr zu klein wäre']
    ),
    (
        'Ein belangloser Satz. Ein Satz mit drei Ausrufungszeichen am Ende!',
        ['Ein belangloser Satz.',
         'Ein Satz mit drei Ausrufungszeichen am Ende!']
    ),
    (
        'Nach ca. 3-4 Monaten (des Dauertragens und -waschens!) kommen '
        'die Bügel raus',
        ['Nach ca. 3-4 Monaten (des Dauertragens und -waschens!) kommen '
         'die Bügel raus']
    ),
    (
        'D. h. man sollte es mal testen.',
        ['D. h. man sollte es mal testen.']
    ),
    (
        'Schöne Grüße von Ursula u. Hans',
        ['Schöne Grüße von Ursula u. Hans']
    ),
    (
        'Das Material ist o. k. - für den Preis wirklich in Ordnung.',
        ['Das Material ist o. k. - für den Preis wirklich in Ordnung.']
    ),
    (
        '5.1 war bei uns nicht gewünscht wegen Kabel, Aufwand, und Optik.',
        ['5.1 war bei uns nicht gewünscht wegen Kabel, Aufwand, und Optik.']
    ),
]


class TestSentenizerTestCases:
    @pytest.fixture
    def sentenize(self):
        from dstoolbox.text import sentenize
        return sentenize

    def test_sentenize_test_cases(self, sentenize):
        """Test all test cases and only fail if more than MAXFAILS
        tests fail.

        """
        maxfails = 0.5
        passes, fails = 0, 0
        excs = []

        for i, (sentences, expected) in enumerate(TEST_CASES):
            result = sentenize(sentences)
            try:
                assert result == expected
                passes += 1
            except AssertionError as exc:
                print('\nFailing:')
                print(' result:   {}'.format(result))
                print(' expected: {}'.format(expected))
                fails += 1
                excs.append((i, exc))

        print('TestSentenizer: {} of {} ({:.2f} %) of tests passed.'.format(
            passes, (passes + fails), passes / (passes + fails) * 100.0
        ))

        if fails / (passes + fails) >= maxfails:
            raise AssertionError(excs)

    def test_sentenize_speed(self, sentenize):
        max_time = 1e-4
        tic = time.time()
        for sentence, _ in TEST_CASES:
            sentenize(sentence)
        toc = time.time()
        time_per_sentence = (toc - tic) / len(TEST_CASES)

        assert time_per_sentence < 1e-4
        print("Average time per sentence: {:.3f} ms ({:.1f} ms max).".format(
            1000 * time_per_sentence, 1000 * max_time))


class TestNormalizeWord:
    @pytest.fixture
    def normalize_word(self):
        from dstoolbox.text import normalize_word
        return normalize_word

    @pytest.mark.parametrize('sentence, expected', [
        ('', ''),
        ('hi', 'hi'),
        (' hi', 'hi'),
        ('hi ', 'hi'),
        (' hi ', 'hi'),
        ('Hi', 'hi'),
        ('hi?', 'hi'),
        ('häßlich', 'häßlich'),
        ('(oder)', 'oder'),
        ('was?', 'was'),
        ('ha!', 'ha'),
        ('"gut"', 'gut'),
        (' "HäßLich?" ', 'häßlich'),
    ])
    def test_normalize_correct_result(
            self, normalize_word, sentence, expected):
        assert normalize_word(sentence) == expected
