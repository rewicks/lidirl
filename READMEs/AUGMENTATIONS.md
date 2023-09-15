# Augmentations

This is a comprehensive list of augmentations that we implemented as well as a brief description and an example. There are currently 15.

## Short Text


This augmentation grabs a span from the input based on length (number of white-space separated words). It does not currently support character-level grabbing (i.e., for Chinese, etc), but this could easily be added.

Right now we're just grabbing length 1--so this is grabbing a random word from a sentence.

Example:

```
from augmentations import Short

aug = Short()
input = "This is an example."
print(aug(input))
```

Output:
```
an
```



## AntSpeak

This augmentation comes from one of the survey papers (Caswell et al. 2020). They have antspeak as inserting whitespace between characters instead of between words. We generalize it as inserting a "special character" between the original characters.

We also optionally shorten the original input. Shortening by default is 4 (white-space separated) words

Example:

```
from augmentations import Antspeak
aug = Antspeak(
                shorten_ratio = 0.0, # this is how often to shorten the augmentation
                special_character = " " # this does the typical antspeak
                    )
input = "This is an example"
print(aug(input))
```

Output:
```
T h i s i s a n e x a m p l e
```

or

```
from augmentations import Antspeak
aug = Antspeak(
                shorten_ratio = 1.0,
                special_character = "*"
                    )
input = "This is a much longer example"
print(aug(input))
```

```
m*u*c*h*l*o*n*g*e*r*e*x*m*a*p*l*e*e*x*a*m*p*l*e
```

## NGrams

This is also from the Caswell paper. This is the cliche example since it messes with the n-grams. I can say when applying this to dev tests, it doesn't have as much of a drastic impact as you might think.

The method works by iterating over the characters and repeating between 1-10 times a random percent of the time. Punctuation is not allowed to repeat.

We also optionally shorten this before applying.

```
from augmentations import NGrams
aug = NGrams(
                shorten_ratio = 0.0, # this is how often to shorten the augmentation
                repeat_ratio = 0.05 # how often to choose repeat a character
            )
input = "This is an example"
print(aug(input))
```

Output:
```
This is an exammmmmmple
```

## Spongebob

This is commonly used on social media. I believe it's origin is the Spongebob meme--thus the name. It goes through and randomly chooses upper or lower case for the characters. Also note that this doesn't have an effect on languages without letter casing.

Example:
```
from augmentations import Spongebob
aug = Spongebob()
input = "This is an example"
print(aug(input))
```

Output:
```
ThIs is an EXAmPle
```

## Hashtags

This is also commonly used on social media. This chooses a small span (default 3 words), and concatenates them together with a hashtag in front. Again, this does not consider languages which do not use whitespace.

```
from augmentations import Hashtags
aug = Hashtags(
                length = 3,
                camel_case = 0.5, # how often to camel case the augmentation               
)
input = "This is an example"
print(aug(input))
```

Output:
```
#ThisIsAn
```

## Codeswitch

This is a very simple iteration on this. One version is to concatenate two strings together with a single space between them.

This is an outlier. Most augmentations return one string when called, this returns two. This is because we don't merge two training examples into one, but augment both in reverse.

```
from augmentations import Codeswitch
aug = Codeswitch()
input_a = "This is an example."
input_b = "C'est un exemple."
print(aug(input_a, input_b))
```

Output:
```
("This is an example. C'est un exemple.", "C'est un exemple. This is an example.")
```

## LeetSpeak

This is the same leet speak that you can read more about here (https://en.wikipedia.org/wiki/Leet). It is a method for substitution characters for lookalikes. The common example being "l33t".

This has a predefined set of leetspeak subtitutions which you can find in the `leetspeak.py` file. They are all sourced from wikipedia or other online tables, but some of them are quite obscure.

The augmentation randomly (percent of the time) applies substitutions to characters found in this list.

Example:
```
from augmentations import LeetSpeak
aug = LeetSpeak(change_ratio=0.3)
input = "This is an example."
print(aug(input))
```

Output:
```
Thi§ is ^n £xamp7e.
```

## Cyrillic

This is similar to the leetspeak idea, except is the substitution of specifically Cyrillic characters that look like Latin ones

The Cyrillic lookup dictionary can be found in the `cyrillic.py` file.

Example:
```
from augmentations import Cyrillic
aug = Cyrillic(change_ratio=0.75)
input = "This is an example"
print(aug(input))
```

Output:
```
TДis is АЙ eҲДmplЄ
```


## URL

This is part of our non-linguistic input augmentations. It ignores the input and instead generates a random url.

Example:
```
from augmentations import URL
aug = URL()
input = "This is an example"
print(aug(input))
```

Output:
```
https://www.PW4uczx1xa88.com/lRFx6qMXp-
```

## HTML

This adds HTML tags (`<b>`/`</b>`, `<i>`/`</i>`, `<u>`/`</u>`, `'<a href=>`/`</a>`) to random spans of the input. For the last case, it also generates a URL (as seen above) to add in.

Example:
```
from augmentations import HTML
aug = HTML()
input = "This is an example"
print(aug(input))
```

Output:
```
This is <a href=http://www.yjK2Uk9s0g.com/U3QQUnL7> an </a> example
```

## Emojis

When we refer to emojis here, we're mostly referring to emoticons (i.e., the types of images that are created through creative use of character sets that are typically used for language) and not the character set that unicode reserves for actual emojis.

The emoticon set can be found in the `emoticons.py` file.

### Replacement

In one case, we can ignore the input text and completely replace with the emoticon. This is another example of the non-linguistic features.

Example:
```
from augmentations import ReplaceEmoji
aug = ReplaceEmoji()
input = "This is an example"
print(aug(input))
```

Output:
```
⊂(◉‿◉)つ
```

### Addition

Alternatively, we can add the emoji to the original input. This will insert it randomly between words (whitespace delimited).

Example:
```
from augmentations import AddEmoji
aug = AddEmoji()
input = "This is an example"
print(aug(input))
```

Output
```
This ╭(ʘ̆~◞౪◟~ʘ̆)╮ is an example
```

## Typo Approximates

We did investigate some more complicated typo augmentations--looking at keymaps so we could do neighbor-substitution. We abandon this because of the effort v. payoff, but let us know if you'd like us to take another look.

### Character Deletion

As it seems--we randomly delete some percent of characters from the input.

Example:
```
from augmentations import Delete
aug = Delete(change_ratio=0.1)
input = "This is an example"
print(aug(input))
```

Output:
```
This is aexample
```

### Character Addition

Also as it seems. A smarter, more accurate version of this is to only add characters that are physically near the same key, but we do something similar. We choose a random character from the input string, and add the character that is 5 away from that same unicode value.

Example:
```
from augmentations import Add
aug = Add(change_ratio=0.3)
input = "This is an example"
print(aug(input))
```

Output:
```
This is afnj emxample
```

### Swap Characters

Randomly swaps two adjacent characters.

Example:
```
from augmentations import Swap
aug = Swap(change_ratio=0.1)
input = "This is an example"
print(aug(input))
```

Output:
```
hTis is a nexample
```


# Adding Augmentations

1. The augmentation to the `augmentations.py` file.
2. Add appropriate logic to `build_augmentations` function in `trainer.py`
3. [optional] Add appropriate logic to `get_batch` in `TrainingShard` class in `preprocessor.py`