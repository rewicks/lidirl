#!/usr/bin/env python3

"""
Inserts different types of noise to data.

"""
from typing import Optional, Any, Union, Callable, Tuple, List, Dict
from .leetspeak import LEETSPEAK
from .cyrillic import CYRILLIC
from .emoticons import EMOTICONS

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

if (__package__ is None or __package__ == "") and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'lidirl'


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

#################################### IMPORTS ####################################

import random
import string
import json

#################################### FUNCTIONALITY ####################################


def build_augmentations(args):
    if args.augmentations is not None:
        augs = []
        probs = []

        for aug in args.augmentations.split('/'):
            aug = aug.split(',')
            prob = float(aug[1])
            aug = aug[0]
            if aug == "antspeak":
                aug = Antspeak()
            elif aug == "ngrams":
                aug = NGrams()
            elif aug == "hashtag":
                aug = Hashtags()
            elif aug == "short":
                aug = Short()
            elif aug == "spongebob":
                aug = Spongebob()
            elif aug == "codeswitch":
                aug = Codeswitch()
            elif aug == "leetspeak":
                aug = LeetSpeak()
            elif aug == "cyrillic":
                aug = Cyrillic()
            elif aug == "url":
                aug = URL()
            elif aug == "html":
                aug = HTML()
            elif aug == "addemojis":
                aug = AddEmoji()
            elif aug == "replaceemojis":
                aug = ReplaceEmoji()
            elif aug == "delete":
                aug = Delete()
            elif aug == "add":
                aug = Add()
            elif aug == "swap":
                aug = Swap()
            else:
                logger.error(f"{aug} not supported")
                exit(-1)

            augs.append(aug)
            probs.append(prob)
        total_prob_mass = sum(probs)
        augmentations = []
        for a, p in zip(augs, probs):
            augmentations.append((a, p/total_prob_mass))
        return augmentations
    return None

def grab_a_span(text : List[int],
                    span_size : int = 3) -> List[int]:
    """
    From a given input, grabs a short span and returns. I think it makes sense to also add some logic for non-whitespace scripts

    :param text: The list of input ids (corresponding to characters of bytes). 
                    Expected to be semi-processed for training (i.e. already mapping to ids instead of input tokens)
    :span_size: The number of words (or equivalent) to include in span. Defaults to 3.
    """

    # count words and determine if there are no space symbols
    num_words = len(text.split())
    # num_words, space_idx = count_words(text, space_idx)

    # if the text is already shorter than our desired span, return
    if num_words < span_size + 1:
        return text

    # randomly chosen index on where to start span
    start_index = random.choice([_ for _ in range(num_words-span_size)])

    # iterate over text to accumulate words in span
    out = []
    index = 0
    for t in " ".join(text.split()):

        # words are counted via space symbol unless there isn't one
        if t == " " or num_words == 1:
            index += 1
        
        # append all words between
        if index >= start_index and index <= start_index+span_size:
            if index == start_index and t != " ":
                out.append(t)
            elif index > start_index:
                out.append(t)
    return out

def count_words(text : List[int], space_idx : int) -> Tuple[int, int]:
    # get total number of words (or tokens if no space symbol)
    num_words = sum([1 if t == space_idx else 0 for t in text]) + 1

    # if there's only one word, no space symbols were found. 
    # assumes all tokens should be considered words.
    if num_words == 1:
        space_idx = None
        num_words = len(text)

    return num_words, space_idx

def byte_to_char(text : List[int]):
    """
        Full disclaimer that this is pretty inefficient. For byte-level inputs, need to map back to chars in order to 
        insert character-level noise. Very unfortunate but necessary for dynamic batch augmentation.

        :param text: the input list of byte ids that would normally be fed to the model
        :param ids_to_bytes: a mapping of the ids to the original bytes (the model vocabulary)

        :return a tuple with:
                - the same input but as characters instead of bytes,
                - the corresponding character_id map
    """

    characters = text.decode('utf-8')
    return characters

def char_to_byte(text: List[int]) -> List[int]:
    """
        Reverses the byte_to_char by converting back to bytes. This is done after inserting noise and before returning it to trainer.

        :param text: the input list of character ids that have been augmented with noise
        :char_ids: the dictionary with how to reverse bytes back to the ids (model input)

        :return augmented byte-level input for model
    """
    return text.encode('utf-8')


class Augmentation():
    """
        A superclass to account for all types of augmentations. All augmentations should be callable.
    """
    def __init__(self):
        pass

    def __call__(self, text):
        return text

class Short():
    """
        Derives short text from longer text.
    """

    def __init__(self, 
                    length : int=1,
                    is_byte : bool = False):
        """
            Initialization function.

            :param length: The length of the span to grab.
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.length = length
        self.is_byte = is_byte

    def __call__(self, text : List[int]) -> List[int]:
        """
            Extracts short text from a longer string of text.

            :param text: the input list of character ids that have been augmented with noise.
            :param length: how long the extracted span should be. Defaults to one word/token

            :return the same input with inserted noise
        """

        # if input is in bytes, convert to chars
        if self.is_byte:
            text = byte_to_char(text)

        # grab a random span of specified length
        out = grab_a_span(text, span_size=self.length)

        out = "".join(out)

        # if input was bytes, convert back
        if self.is_byte:
            out = char_to_byte(out)

        return out


class Antspeak(Augmentation):
    """
        Inserts a new special character (typically a space) between each normal character.
    """

    def __init__(self, 
                    is_byte : bool = False,
                    shorten_ratio : float = 0.5,
                    special_character : str = " "):
        """
            Initialization function

            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param shorten_ratio: The ratio at which to grab a span for antspeak instead of using the whole thing.
            :param special_character: The character to insert. Default is " ".
        """
        self.is_byte = is_byte
        self.shorten_ratio = shorten_ratio
        self.special_character = special_character

    def __call__(self, text : List[int]):
        """
            Inserts special character between original characters.

            :param text: the input list of character ids that have been augmented with noise.

            :return the same input with inserted noise
        """
        # if input is in bytes, convert to chars
        if self.is_byte:
            text = byte_to_char(text)

        if random.random() < self.shorten_ratio:
            text = grab_a_span(text)

        out = []
        for t in text[:-1]:
            if t != " ":
                out.append(t)
                out.append(self.special_character if self.special_character is not None else " ")
        out.append(text[-1])

        out = "".join(out)

        # if input was bytes, convert back
        if self.is_byte:
            out = char_to_byte(out)

        return out

class NGrams(Augmentation):
    """
        Inserts noise by repeating n-grams.
    """

    def __init__(self,
                    is_byte : bool = False, 
                    shorten_ratio : float = 0.5,
                    repeat_ratio : float = 0.05):
        """
            Initialization function

            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param shorten_ratio: The percent at which to grab a span for antspeak instead of using the whole thing.
            :param repeat_ratio: The percent of time that a character is repeated. Default 0.05.
        """
        self.is_byte = is_byte
        self.repetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.shorten_ratio = shorten_ratio
        self.repeat_ratio = repeat_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly repeats some number of characters in input text.

            :param text: the input list of character ids that have been augmented with noise.

            :return the same input with inserted noise
        """
        # if input is bytes, reverse to chars
        if self.is_byte:
            text = byte_to_char(text)

        if random.random() < self.shorten_ratio:
            text = grab_a_span(text)

        out = []
        repeated = False
        for i, t in enumerate(text):
            out.append(t)
            if not repeated and t not in string.punctuation + " " and i == len(text) - 1 or random.random() <= 0.05:
                repeated = True
                repeat = random.choice(self.repetitions)
                for r in range(repeat):
                    out.append(t)

        out = "".join(out)
        # if input was bytes, convert back to bytes
        if self.is_byte:
            out = char_to_byte(out)

        return out

class Spongebob(Augmentation):
    """
        Alternates case of input string in a fashion similar to "spongebob meme font"
    """

    def __init__(self, 
                    is_byte : bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.is_byte = is_byte

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly alters case between upper and lower case (with slight preference towards lower case)

            :param text: the input list of character ids that have been augmented with noise.

            :return the same input with inserted noise
        """

        # if input is bytes, reverse to chars
        if self.is_byte:
            text = byte_to_char(text)

        out = []
        for t in text:
            if random.random() <= 0.5:
                out.append(t.upper())
            else:
                out.append(t.lower())

        out = "".join(out)

        # if input was bytes, convert back to bytes
        if self.is_byte:
            out = char_to_byte(out)

        return out

class Hashtags(Augmentation):
    """
        Randomly extracts a span and inserts a hashtag.
    """

    def __init__(self,
                    length : int = 3,
                    camel_case : float = 0.5,
                    is_byte : bool = False):
        """
            Initialization function

            :param length: the length of the extracted span to use
            :param camel_case: how often to turn the extracted span into camel case instead of all lower case
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.length = length
        self.camel_case = camel_case
        self.is_byte = is_byte
        self.special_character = '#'

    def get_span(self, 
                    text : List[int], 
                    start : int,
                    end : int,
                    camel_case : bool = False) -> List[int]:
        """
            From a given input, grabs a short span and returns. Assumes words are separated by "space_idx".
            This only differs from other function because of the camel case/lower case feature.

            :param text: The list of input ids (corresponding to characters of bytes). 
                            Expected to be semi-processed for training (i.e. already mapping to ids instead of input tokens)
            :param start: Index to start at
            :param end: Index to end at
            :param camel_case determines if the returned hashtag will be camel case or not

            :return extracted span in appropriate lower/camel case format
        """
        out = []
        index = 0
        word_begin = True
        for t in text:
            if t == " ":
                index += 1
                word_begin = True
            else:
                if index >= start and index <= end:
                    if t not in string.punctuation:
                        if camel_case and word_begin:
                            out.append(t.upper())
                        else:
                            out.append(t.lower())
                        word_begin = False
        return out
        
    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly selects a span and puts a hashtag in front of it
        """

        # if input is bytes, reverse to chars
        if self.is_byte:
            text = byte_to_char(text)

        num_words = len(text.split(" "))

        if num_words <= self.length:
            index = 0
        else:
            index = random.choice([_ for _ in range(num_words-self.length)])
        camel_case = random.random() < self.camel_case
        out = [self.special_character] + self.get_span(text, index, index+self.length-1, camel_case=camel_case)

        out = "".join(out)

        # if input was bytes, convert back to bytes
        if self.is_byte:
            out = char_to_byte(out)

        return out

class Codeswitch(Augmentation):
    """
        Returns the concatenation of a+b and b+a
    """

    def __init__(self,
                    is_byte: bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.space_span = [" "]
        self.is_byte = is_byte
        
    def __call__(self, text : List[int], switch : List[int]) -> Tuple[List[int], List[int]]:
        """
            Returns the concatenation of a+b and b+a
        """
        if self.is_byte:
            text = byte_to_char(text)
            switch = byte_to_char(switch)

        text = [_ for _ in text]
        switch = [_ for _ in switch]

        out = "".join(text + self.space_span + switch), "".join(switch + self.space_span + text)

        if self.is_byte:
            out = char_to_byte(out[0]), char_to_byte(out[1])

        return out

class LeetSpeak(Augmentation):
    """
        Randomly substitutes Latin characters for a lookalike.
    """

    def __init__(self,
                    is_byte: bool = False,
                    change_ratio : float = 0.3):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param change_ratio: How often to change a character that is found in the leetspeak dictionary
        """
        self.is_byte = is_byte
        self.change_ratio = change_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly substitutes Latin characters for a lookalike.
        """
        if self.is_byte:
            text = byte_to_char(text)

        
        out = []
        for t in text:
            if t in LEETSPEAK and random.random() < self.change_ratio:
                out.append(random.choice(LEETSPEAK[t]))
            else:
                out.append(t)

        out = "".join(out)

        if self.is_byte:
            out = char_to_byte(out)

        return out

class Cyrillic(Augmentation):
    """
        Randomly substitutes Latin characters for Cyrillic lookalikes.
    """

    def __init__(self,
                    is_byte: bool = False,
                    change_ratio : float = 0.75):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param change_ratio: How often to change a character that is found in the Cyrillic dictionary
        """
        self.is_byte = is_byte
        self.change_ratio = change_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly substitutes Latin characters for Cyrillic lookalikes.
        """
        if self.is_byte:
            text = byte_to_char(text)

        out = []
        for t in text:
            if t in CYRILLIC and random.random() < self.change_ratio:
                out.append(random.choice(CYRILLIC[t]))
            else:
                out.append(t)

        out = "".join(out)

        if self.is_byte:
            out = char_to_byte(out)

        return out

def generate_url():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-_"
    url = random.choice(["http", "https"])
    url += "://www."
    domain_length = random.choice([5,6,7,8,9,10,11,12])
    for _ in range(domain_length):
        url += random.choice(alphabet)
    url += '.com/'
    postfix_length = random.choice([5,6,7,8,9,10,11,12])
    for _ in range(postfix_length):
        url += random.choice(alphabet)
    return url

class URL(Augmentation):
    """
        Ignores the input and returns a randomly generated URL
    """

    def __init__(self,
                    is_byte: bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.is_byte = is_byte

    def __call__(self, text : List[int]) -> List[int]:
        """
            Ignores the input and returns a randomly generated URL
        """

        url = generate_url()
        if self.is_byte:
            url = char_to_byte(url)
        return url

class HTML(Augmentation):
    """
        Randomly adds a small set of HTML tags to the input string.
    """

    def __init__(self,
                    is_byte: bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.is_byte = is_byte
        self.tags = [
            ("<b>", "</b>"),
            ("<i>", "</i>"),
            ("<u>", "</u>"),
            ("<s>", "</s>"),
            ("<a href=", "</a>"),
        ]


    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly adds a small set of HTML tags to the input string.
        """
        if self.is_byte:
            text = byte_to_char(text)

        words = text.split(" ")
        if len(words) > 1:
            start_index = random.choice([_ for _ in range(len(words)-1)])
            end_index = start_index + random.choice([1, 2, 3])
            if end_index >= len(words):
                end_index = len(words) - 1
            
            tags = random.choice(self.tags)
            start_tag = tags[0]
            end_tag = tags[1]
            if "a href" in tags[0]:
                start_tag += generate_url() + ">"
            out = " ".join(words[:start_index] + [start_tag] + words[start_index:end_index] + [end_tag] + words[end_index:])
        else:
            return text

        out = "".join(out)

        if self.is_byte:
            out = char_to_byte(out)

        return out

class ReplaceEmoji(Augmentation):
    """
        Ignores the input string, and returns an emoticon.
    """

    def __init__(self,
                    is_byte: bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.is_byte = is_byte


    def __call__(self, text : List[int]) -> List[int]:
        """
            Ignores the input string, and returns an emoticon.
        """
        out = random.choice(EMOTICONS)
        if self.is_byte:
            out = char_to_byte(out)
        return out

class AddEmoji(Augmentation):
    """
        Randomly adds an emoticon between words in the input string.
    """

    def __init__(self,
                    is_byte: bool = False):
        """
            Initialization function
            :param is_byte: If training data is byte level or character level. Default: false (character level)
        """
        self.is_byte = is_byte

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly adds an emoticon between words in the input string.
        """

        emoticon = random.choice(EMOTICONS)

        if self.is_byte:
            text = byte_to_char(text)

        words = text.split(" ")
        insertion_index = random.choice([_ for _ in range(len(words))])
        if insertion_index == 0:
            out = emoticon + " " + text
        else:
            out = " ".join(words[:insertion_index] + [emoticon] + words[insertion_index:])

        if self.is_byte:
            out = char_to_byte(out)
        return out


class Delete(Augmentation):
    """
        Randomly deletes a character from the input string.
    """

    def __init__(self,
                    is_byte: bool = False,
                    change_ratio : float = 0.1):
        """
            Initialization function
            :param is_byte: Whether the input is a byte string or not
            :param change_ratio: The probability of deleting a character
        """
        self.is_byte = is_byte
        self.change_ratio = change_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly deletes a character from the input string.
        """
        if self.is_byte:
            text = byte_to_char(text)

        
        out = []
        for t in text:
            if random.random() >= self.change_ratio:
                out.append(t)

        out = "".join(out)

        if len(out) == 0:
            out = text

        if self.is_byte:
            out = char_to_byte(out)

        return out

class Swap(Augmentation):
    """
        Randomly swaps the position of two adjacent characters in the input string.
    """

    def __init__(self,
                    is_byte: bool = False,
                    change_ratio : float = 0.1):
        """
            Initialization function
            :param is_byte: Whether the input is a byte string or not
            :param change_ratio: The probability that a character will be swapped
        """
        self.is_byte = is_byte
        self.change_ratio = change_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly swaps the position of two adjacent characters in the input string.
        """
        if self.is_byte:
            text = byte_to_char(text)

        out = []
        for t in text:
            if random.random() >= self.change_ratio or len(out) == 0:
                out.append(t)
            else:
                out = out[:-1] + [t] + [out[-1]]
                
        out = "".join(out)


        if self.is_byte:
            out = char_to_byte(out)

        return out


class Add(Augmentation):
    """
        Randomly adds in new character to input string
    """

    def __init__(self,
                    is_byte: bool = False,
                    change_ratio : float = 0.3):
        """
            Initialization function
            :param is_byte: Whether the input is a byte string or not
            :param change_ratio: The ratio of characters to change
        """
        self.is_byte = is_byte
        self.change_ratio = change_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly adds in new character to input string
        """
        if self.is_byte:
            text = byte_to_char(text)

        out = []
        for t in text:
            if random.random() < self.change_ratio:
                new_char = random.choice([ord(_) for _ in text])
                new_char += 5
                try:
                    out.append(chr(new_char))
                except:
                    pass
            out.append(t)
                
        out = "".join(out)

        if self.is_byte:
            out = char_to_byte(out)

        return out
