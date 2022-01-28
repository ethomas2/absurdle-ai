from dataclasses import dataclass
import enum
import typing as t
# import functools
import itertools


class Player(enum.Enum):
    HUMAN = enum.auto()
    ABSURDLE = enum.auto()


class HintItem(enum.Enum):
    GREEN = enum.auto()
    YELLOW = enum.auto()
    BLACK = enum.auto()


Word = str

Hint = t.List[HintItem]


@dataclass
class State:
    # TODO: validate pydantic if errors
    # - all words are from word list
    # - if player  == human -> len(words) == len(hints)
    # - if player == absurdle -> len(words) == len(hints) + 1
    player: Player
    words: t.List[Word]
    hints: t.List[Hint]

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


def _next_possible_words(state: State) -> t.Iterator[Word]:
    """
    actions that PLAYER can take
    given state what are all the possible words that could fit the hints
    """
    assert state.player == Player.HUMAN
    assert len(state.words) == len(state.hints)

    for potential_word in WORD_LIST:
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
        exists = any(
            _is_consistent_history(true_word, state.words, state.hints + [hint])
            for true_word in WORD_LIST
        )
        if exists:
            yield hint


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
        return min(
            (minimax(next_state) for next_state in next_possible_states),
            key=len,
        )

    elif state.player == Player.ABSURDLE:
        next_possible_states = (
            State(Player.HUMAN, state.words, state.hints + [hint])
            for hint in _next_possible_hints(state)
        )
        return max(
            (minimax(next_state) for next_state in next_possible_states),
            key=len,
        )
    else:
        raise Exception(
            "Non exhaustive enum (player should be HUMAN or ABSURDLE)"
        )


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
