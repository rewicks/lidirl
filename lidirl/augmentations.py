#!/usr/bin/env python3

"""
Inserts different types of noise to data.

"""
from typing import Optional, Any, Union, Callable, Tuple, List, Dict

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

if __package__ is None and __name__ == '__main__':
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

#################################### FUNCTIONALITY ####################################

import random

def grab_a_span(text : List[int], 
                    space_idx : int,
                    span_size : int = 3) -> List[int]:
    """
    From a given input, grabs a short span and returns. Assumes words are separated by "space_idx".

    :param text: The list of input ids (corresponding to characters of bytes). 
                    Expected to be semi-processed for training (i.e. already mapping to ids instead of input tokens)
    :param space_idx: The value of the space character for separating words. 
                        If this character is not found in the text, it assumes words are not whitespace separated (used for Chinese or similar).
    :span_size: The number of words (or equivalent) to include in span. Defaults to 3.
    """

    # count words and determine if there are no space symbols
    num_words, space_idx = count_words(text, space_idx)

    # if the text is already shorter than our desired span, return
    if num_words < span_size + 1:
        return text

    # randomly chosen index on where to start span
    start_index = random.choice([_ for _ in range(num_words-span_size)])

    # iterate over text to accumulate words in span
    out = []
    index = 0
    for t in text:

        # words are counted via space symbol unless there isn't one
        if t == space_idx or space_idx is None:
            index += 1
        
        # append all words between
        if index >= start_index and index <= start_index+span_size:
            if index == start_index and t != space_idx:
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

def byte_to_char(text : List[int], ids_to_bytes: Dict) -> Tuple[List[int], Dict]:
    """
        Full disclaimer that this is pretty inefficient. For byte-level inputs, need to map back to chars in order to 
        insert character-level noise. Very unfortunate but necessary for dynamic batch augmentation.

        :param text: the input list of byte ids that would normally be fed to the model
        :param ids_to_bytes: a mapping of the ids to the original bytes (the model vocabulary)

        :return a tuple with:
                - the same input but as characters instead of bytes,
                - the corresponding character_id map
    """

    byte_string = []
    bytes_to_ids = {}
    for t in text:
        byte = ids_to_bytes[t]
        byte_string.append(byte)
        bytes_to_ids[byte] = t

    characters = bytes(byte_string).decode('utf-8')
    
    out = []
    for c in characters:
        out.append(ord(c))
        
    return (bytes_to_ids, bytes_to_ids)

def char_to_byte(text: List[int], bytes_to_ids: Dict) -> List[int]:
    """
        Reverses the byte_to_char by converting back to bytes. This is done after inserting noise and before returning it to trainer.

        :param text: the input list of character ids that have been augmented with noise
        :char_ids: the dictionary with how to reverse bytes back to the ids (model input)

        :return augmented byte-level input for model
    """

    char_string = []
    for t in text:
        char_string.append(chr(t))
    
    char_string = "".join(char_string).encode('utf-8')

    out = []
    for c in char_string:
        out.append(bytes_to_ids[c])

    return out

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
                    space_idx : int,
                    length : int=1,
                    is_byte : bool = False, 
                    byte_ids : Dict = None):
        """
            Initialization function.

            :param space_idx: The id for the space symbol in the vocabulary.
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param byte_ids: Model's vocabulary to reverse ids back to bytes.
        """
        self.space_idx = space_idx
        self.length = length

        self.is_byte = is_byte
        if is_byte and byte_ids is None:
            raise RuntimeError("is_byte set to True but byte_ids not provided.")
        self.byte_ids = byte_ids



    def __call__(self, text : List[int]) -> List[int]:
        """
            Extracts short text from a longer string of text.

            :param text: the input list of character ids that have been augmented with noise.
            :param length: how long the extracted span should be. Defaults to one word/token

            :return the same input with inserted noise
        """

        # if input is in bytes, convert to chars
        if self.is_byte:
            text, map = byte_to_char(text, self.byte_ids)

        # grab a random span of specified length
        out = grab_a_span(text, space_idx=self.space_idx, span_size=self.length)

        # if input was bytes, convert back
        if self.is_byte:
            out = char_to_byte(out, map)

        return out


class Antspeak(Augmentation):
    """
        Inserts a new special character (typically a space) between each normal character.
    """

    def __init__(self, 
                    space_idx : int,
                    is_byte : bool = False,
                    byte_ids : Dict = None,
                    shorten_ratio : float = 0.5,
                    special_character : int = None):
        """
            Initialization function

            :param space_idx: The id for the space symbol in the vocabulary.
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param byte_ids: Model's vocabulary to reverse ids back to bytes.
            :param shorten_ratio: The ratio at which to grab a span for antspeak instead of using the whole thing.
            :param special_character: The character to insert. If None, insert space_idx.
        """
        self.space_idx = space_idx
        self.is_byte = is_byte
        if is_byte and byte_ids is None:
            raise RuntimeError("is_byte set to True but byte_ids not provided.")
        self.byte_ids = byte_ids
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
            text, map = byte_to_char(text, self.byte_ids)

        if random.random() < self.shorten_ratio:
            text, _ = grab_a_span(text, space_idx=self.space_idx)
        out = []
        for t in text[:-1]:
            out.append(t)
            out.append(self.special_charater if self.special_character is not None else self.space_idx)
        out.append(text[-1])

        # if input was bytes, convert back
        if self.is_byte:
            out = char_to_byte(out, map)

        return out

class NGrams(Augmentation):
    """
        Inserts noise by repeating n-grams.
    """

    def __init__(self, 
                    disallowed_repeats : List[int],
                    space_idx : int, 
                    is_byte : bool = False, 
                    byte_ids : Dict = None,
                    shorten_ratio : float = 0.5):
        """
            Initialization function

            :param disallowed_repeats: a list of tokens that shouldn't be repeated. Reserved for punctuation and similar tokens.
            :param space_idx: The id for the space symbol in the vocabulary.
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param byte_ids: Model's vocabulary to reverse ids back to bytes.
            :param shorten_ratio: The ratio at which to grab a span for antspeak instead of using the whole thing.
            :param special_character: The character to insert. If None, insert space_idx.
        """
        self.disallowed_repeats = disallowed_repeats
        self.space_idx = space_idx
        self.is_byte = is_byte
        if is_byte and byte_ids is None:
            raise RuntimeError("is_byte set to True but byte_ids not provided.")
        self.byte_ids = byte_ids
        self.repetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.shorten_ratio = shorten_ratio

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly repeats some number of characters in input text.

            :param text: the input list of character ids that have been augmented with noise.

            :return the same input with inserted noise
        """

        # if input is bytes, reverse to chars
        if self.is_byte:
            text, map = byte_to_char(text, self.byte_ids)

        if random.random() < self.shorten_ratio:
            text, _ = grab_a_span(text, space_idx=self.space_idx)

        out = []
        repeated = False
        for i, t in enumerate(text):
            out.append(t)
            if not repeated and i == len(text) - 1 or random.random() <= 0.05:
                repeated = True
                repeat = random.choice(self.repetitions)
                for r in range(repeat):
                    out.append(t)

        # if input was bytes, convert back to bytes
        if self.is_bytes:
            out = char_to_byte(out, map)

        return out

class Spongebob(Augmentation):
    """
        Alternates case of input string in a fashion similar to "spongebob meme font"
    """

    def __init__(self, 
                    capitals : Dict, 
                    lowers : Dict, 
                    is_byte : bool = False,
                    byte_ids : Dict = None):
        """
            Initialization function

            :param capitals: a dictionary of every token to its upper-case version
            :param lowers: a dictionary of every token to its lower-case version
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param byte_ids: Model's vocabulary to reverse ids back to bytes.
        """
        self.capitals = capitals
        self.lowers = lowers
        self.is_byte = is_byte
        if is_byte and byte_ids is None:
            raise RuntimeError("is_byte set to True but byte_ids not provided.")
        self.byte_ids = byte_ids

    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly alters case between upper and lower case (with slight preference towards lower case)

            :param text: the input list of character ids that have been augmented with noise.

            :return the same input with inserted noise
        """

        # if input is bytes, reverse to chars
        if self.is_byte:
            text, map = byte_to_char(text, self.byte_ids)

        out = []
        for t in text:
            if random.random() <= 0.4:
                out.append(self.capitals.get(t, t))
            else:
                out.append(self.lowers.get(t, t))

        # if input was bytes, convert back to bytes
        if self.is_bytes:
            out = char_to_byte(out, map)

        return out

class Hashtags(Augmentation):
    """
        Randomly extracts a span and inserts a hashtag.
    """

    def __init__(self, 
                    hashtag_idx : int, 
                    space_idx : int, 
                    punctuation : List[int],
                    capitals : Dict,
                    lowers : Dict,
                    length : int = 3,
                    camel_case : float = 0.5,
                    is_byte : bool = False,
                    byte_ids : Dict = None):
        """
            Initialization function

            :param hashtag_idx: id for hashtag symbol in model vocabulary
            :param space_idx: id for space symbol in model vocabulary
            :param punctuation: a list of all ids for punctuation
            :param capitals: a dictionary of every token to its upper-case version
            :param lowers: a dictionary of every token to its lower-case version
            :param length: the length of the extracted span to use
            :param camel_case: how often to turn the extracted span into camel case instead of all lower case
            :param is_byte: If training data is byte level or character level. Default: false (character level)
            :param byte_ids: Model's vocabulary to reverse ids back to bytes.
        """
        self.hashtag_idx = hashtag_idx
        self.space_idx = space_idx
        self.punctuation=punctuation
        self.capitals=capitals
        self.lowers=lowers
        self.length = length
        self.camel_case = camel_case
        self.is_byte = is_byte
        if is_byte and byte_ids is None:
            raise RuntimeError("is_byte set to True but byte_ids not provided.")
        self.byte_ids = byte_ids

    def get_span(self, 
                    text : List[int], 
                    start : int,
                    end : int,
                    space_idx : int = None,
                    camel_case : bool = False) -> List[int]:
        """
            From a given input, grabs a short span and returns. Assumes words are separated by "space_idx".
            This only differs from other function because of the camel case/lower case feature.

            :param text: The list of input ids (corresponding to characters of bytes). 
                            Expected to be semi-processed for training (i.e. already mapping to ids instead of input tokens)
            :param start: Index to start at
            :param end: Index to end at
            :param space_idx: The value of the space character for separating words. 
                                If this character is not found in the text, it assumes words are not whitespace separated (used for Chinese or similar).
            :param camel_case determines if the returned hashtag will be camel case or not

            :return extracted span in appropriate lower/camel case format
        """
        out = []
        index = 0
        word_begin = True
        for t in text:
            if t == space_idx or space_idx is None:
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
        
    def __call__(self, text : List[int]) -> List[int]:
        """
            Randomly selects a span and puts a hashtag in front of it
        """

        # if input is bytes, reverse to chars
        if self.is_byte:
            text, map = byte_to_char(text, self.byte_ids)

        num_words, space_idx = count_words(text, self.space_idx)

        if num_words <= self.length:
            index = 0
        else:
            index = random.choice([_ for _ in range(num_words-self.length)])
        camel_case = random.random() < self.camel_case_ratio
        out = [self.hashtag_idx] + self.get_span(text, index, index+self.length-1, space_idx=space_idx, camel_case=camel_case)

        # if input was bytes, convert back to bytes
        if self.is_bytes:
            out = char_to_byte(out, map)

        return out
