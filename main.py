from dataclasses import dataclass
import collections
import enum
import functools
import itertools
import typing as t


class Player(enum.Enum):
    HUMAN = enum.auto()
    ABSURDLE = enum.auto()

    def __str__(self):
        if self == Player.HUMAN:
            return "human"
        elif self == Player.ABSURDLE:
            return "absurdle"


class HintItem(enum.Enum):
    GREEN = enum.auto()
    YELLOW = enum.auto()
    BLACK = enum.auto()

    @property
    def char(self):
        return {
            HintItem.GREEN: "G",
            HintItem.YELLOW: "Y",
            HintItem.BLACK: "B",
        }[self]


Word = str

Hint = t.List[HintItem]
HashableHint = t.Tuple[HintItem]


@dataclass
class State:
    # TODO: validate pydantic if errors
    # - all words are from word list
    # - if player  == human -> len(words) == len(hints)
    # - if player == absurdle -> len(words) == len(hints) + 1
    player: Player
    words: t.List[Word]
    hints: t.List[Hint]

    # _possible_words: t.List[Word]

    @property
    def is_terminal_node(self):
        return self.hints and all(
            hint_item == HintItem.GREEN for hint_item in self.hints[-1]
        )


WORD_LENGTH = 5


def _get_word_list() -> t.List[Word]:
    with open("word-list.txt") as f:
        words = [line.strip() for line in f.readlines() if line.strip()]
    assert all(len(w) == WORD_LENGTH for w in words)
    return words


WORD_LIST = _get_word_list()


def _matches(potential_word: Word, guessed_word: Word, hint: Hint) -> bool:
    """
    Returns if the word <potential_word> is possible given the hint
    (guessed_word, hint) was already given
    """
    for (i, (ch, hint_item)) in enumerate(zip(guessed_word, hint)):
        if hint_item == HintItem.GREEN and potential_word[i] != ch:
            return False
        elif hint_item == HintItem.BLACK and ch in potential_word:
            return False
        elif hint_item == HintItem.YELLOW and (
            potential_word[i] == ch or ch not in potential_word
        ):
            return False
    return True


@functools.cache
def _narrow_wlist(hashable_hint: HashableHint, word: str) -> t.Set[Word]:
    """ seems to work. Dramatically reduces word list (2000 -> 500). Not good enough """
    return set(
        output_word
        for output_word in WORD_LIST
        if _matches(output_word, word, list(hashable_hint))
    )


def _next_possible_words(state: State) -> t.Iterator[Word]:
    """
    actions that HUMAN can take
    given state what are all the possible words that could fit the hints
    """
    assert state.player == Player.HUMAN
    assert len(state.words) == len(state.hints)

    reduced_word_list = functools.reduce(
        lambda a, b: a & b,
        (
            _narrow_wlist(tuple(hint), word)
            for (word, hint) in zip(state.words, state.hints)
        ),
    )
    for potential_word in reduced_word_list:
        if all(
            _matches(potential_word, guessed_word, hint)
            for guessed_word, hint in zip(state.words, state.hints)
        ):
            yield potential_word


def _next_possible_hints(state: State) -> t.Iterator[Hint]:
    # return list(map(list, itertools.product(HintItem, repeat=WORD_LENGTH)))

    """
    It's absurdle's turn
    - make a hint possibility
    - run through word list

    - constraint 1: have to pick a hint that doesn't narrow possible words down to 0
    - constraint 2:
        - have to put greens where there were greens,
        - black where there were black
        -

    Have to pick a hint st there exists a word W that makes history consistent

    Input:
        word -- hint
        word -- hint
        word -- hint
        word
    """
    assert state.player == Player.ABSURDLE
    assert len(state.words) == len(state.hints) + 1

    def _is_consistent_history(
        true_word: Word, words: t.List[Word], hints: t.List[Hint]
    ):
        assert len(words) == len(hints)
        return all(_matches(true_word, w, h) for (w, h) in zip(words, hints))

    for hint in map(list, itertools.product(HintItem, repeat=WORD_LENGTH)):

        if len(state.hints) > 0 and any(
            state.hints[-1][i] == HintItem.GREEN and hint[i] != HintItem.GREEN
            for i in range(WORD_LENGTH)
        ):
            # optimization: we know hint has to keep all the greens in place,
            # so discard any that don't
            continue

        exists = any(
            _is_consistent_history(
                true_word, state.words, state.hints + [hint]
            )
            for true_word in WORD_LIST
        )
        if exists:
            yield hint


def _format(state: State) -> str:
    def _format_hint(hint: Hint) -> str:
        return "".join(h.char for h in hint)

    s = ""
    for (i, (word, hint)) in enumerate(zip(state.words, state.hints)):
        s += (
            word
            + "\t"
            + _format_hint(hint)
            + (f"\t{state.player}" if i == 0 else "")
            + "\n"
        )

    if len(state.words) > len(state.hints):
        s += state.words[-1] + "\n"

    return s


_call_stack = -1
nodes_visited = collections.defaultdict(int)


def _debug(f):
    def g(state):
        global _call_stack
        _call_stack += 1
        nodes_visited[_call_stack] += 1
        if _call_stack == 1:
            print(nodes_visited[_call_stack])
            print(_format(state))
        retval = f(state)
        _call_stack -= 1
        return retval

    return g


@_debug
def minimax(state: State) -> t.List[State]:
    """
    minimax(state, player)

        if player == humman
            return min (minimax(possible next states))
        elif player == absurdle
            return argmmax(board, mi

    speedup with oopening book or look up of words * greens -> {set of possible words}
    returns sequence of states to be taken froom this state under optimal play
    """

    if state.is_terminal_node:
        return [state]
    elif state.player == Player.HUMAN:
        next_possible_states = [  # TODO: test out if this is faster with list or generator comprehension
            State(Player.ABSURDLE, state.words + [word], state.hints)
            for word in _next_possible_words(state)
        ]
        result = min(
            (minimax(next_state) for next_state in next_possible_states),
            key=len,
        )
        result = [state] + result

    elif state.player == Player.ABSURDLE:
        next_possible_states = (
            State(Player.HUMAN, state.words, state.hints + [hint])
            for hint in _next_possible_hints(state)
        )
        result = max(
            (minimax(next_state) for next_state in next_possible_states),
            key=len,
        )
        result = [state] + result
    else:
        raise Exception(
            "Non exhaustive enum (player should be HUMAN or ABSURDLE)"
        )

    print(len(result))
    return result


# print(minimax

if __name__ == "__main__":
    """
    main --word <...>

    main --interactive
    >>> HUMAN: ... human enters what he wants ...
    >>> WHAT ABSURD SHOULD DO:  ... write what it thinks absurdle should do
    >>> ABSURDLE: YBYYG
    >>> WHAT HUMAN SHOULD DO: ...
    ...
    >>> FIN
    """

    start_state = State(Player.ABSURDLE, ["adieu"], [])
    print(minimax(start_state))
