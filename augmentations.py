import random

def grab_a_span(text, space_idx):
    num_words = sum([1 if t == space_idx else 0 for t in text]) + 1
    if num_words < 4:
        return text
    start_index = random.choice([_ for _ in range(num_words-3)])
    out = []
    index = 0
    for t in text:
        if t == space_idx:
            index += 1
        if index >= start_index and index <= start_index+3:
            if index == start_index and t != space_idx:
                out.append(t)
            elif index > start_index:
                out.append(t)
    
    return out

class Augmentation():

    def __init__(self):
        pass

    def __call__(self, text):
        return text

class Short():

    def __init__(self, space_idx, is_byte=False):
        self.space_idx = space_idx

    def __call__(self, text):
        num_words = sum([1 if t == self.space_idx else 0 for t in text]) + 1
        word_index = random.choice([_ for _ in range(num_words)])
        out = []
        index = 0
        for t in text:
            if t == self.space_idx:
                index += 1
                if index > word_index:
                    break
            else:
                if index == word_index:
                    out.append(t)
        return out


class Antspeak(Augmentation):

    def __init__(self, space_idx, is_byte=False, shorten_when_possible=True):
        self.space_idx = space_idx
        self.is_byte = is_byte
        self.shorten_when_possible = shorten_when_possible

    def __call__(self, text):
        if random.random() < 0.5:
            text = grab_a_span(text, space_idx=self.space_idx)
        out = []
        for t in text[:-1]:
            out.append(t)
            out.append(self.space_idx)
        out.append(text[-1])
        return out

class NGrams(Augmentation):

    def __init__(self, disallowed_repeats, space_idx, is_byte=False, shorten_when_possible=True):
        self.disallowed_repeats = disallowed_repeats
        self.space_idx = space_idx
        self.is_byte = is_byte
        self.repetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.shorten_when_possible = shorten_when_possible

    def __call__(self, text):
        if random.random() < 0.5:
            text = grab_a_span(text, space_idx=self.space_idx)
        out = []
        repeated = False
        for i, t in enumerate(text):
            out.append(t)
            if not repeated and i == len(text) - 1 or random.random() <= 0.05:
                repeated = True
                repeat = random.choice(self.repetitions)
                for r in range(repeat):
                    out.append(t)
        return out

class Spongebob(Augmentation):

    def __init__(self, capitals, lowers, is_byte=False):
        self.capitals = capitals
        self.lowers = lowers
        self.is_byte = is_byte

    def __call__(self, text):
        out = []
        for t in text:
            if random.random() <= 0.4:
                out.append(self.capitals.get(t, t))
            else:
                out.append(self.lowers.get(t, t))
        return out

class Hashtags(Augmentation):

    def __init__(self, hashtag_idx, space_idx, punctuation, capitals, lowers, is_byte=False):
        self.hashtag_idx = hashtag_idx
        self.space_idx = space_idx
        self.punctuation=punctuation
        self.capitals=capitals
        self.lowers=lowers
        self.is_byte = is_byte

    def get_span(self, text, start, end, camel_case=False):
        out = []
        index = 0
        word_begin = True
        for t in text:
            if t == self.space_idx:
                index += 1
                word_begin = True
            else:
                if index >= start and index <= end:
                    if t not in self.punctuation:
                        if camel_case and word_begin:
                            out.append(self.capitals.get(t, t))
                        else:
                            out.append(self.lowers.get(t,t))
                        word_begin = False
        return out
        
    def __call__(self, text):
        num_words = sum([1 if t == self.space_idx else 0 for t in text]) + 1
        if num_words < 4:
            index = 0
        else:
            index = random.choice([_ for _ in range(num_words-3)])
        camel_case = random.random() < 0.5
        out = [self.hashtag_idx] + self.get_span(text, index, index+2, camel_case=camel_case)
        return out
