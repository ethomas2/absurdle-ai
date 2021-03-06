from dataclasses import dataclass
import collections
import enum
import functools
import itertools
import typing as t
import os


DEBUG = os.environ.get("DEBUG") is not None and os.environ[
    "DEBUG"
].lower() in [
    "yes",
    "true",
    "1",
    "y",
]


class Player(enum.Enum):
    HUMAN = enum.auto()
    ABSURDLE = enum.auto()

    def __str__(self):
        if self == Player.HUMAN:
            return "human"
        elif self == Player.ABSURDLE:
            return "absurdle"


class InvalidHintItem(Exception):
    pass


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

    @classmethod
    def from_char(cls, ch) -> "HintItem":
        result = {
            "G": HintItem.GREEN,
            "Y": HintItem.YELLOW,
            "B": HintItem.BLACK,
        }.get(ch.upper())
        if result is None:
            raise InvalidHintItem(ch)
        return result


Word = str

Hint = t.List[HintItem]
HashableHint = t.Tuple[HintItem]


def _narrow_word_universe(
    universe: t.List[Word], words: t.List[Word], hints: t.List[Hint]
) -> t.List[Word]:
    assert len(words) == len(hints), f"{len(words)=} {len(hints)=}"
    return [
        word
        for word in universe
        if all(_matches(word, w, h) for (w, h) in zip(words, hints))
    ]


@dataclass
class State:
    player: Player
    words: t.List[Word]
    hints: t.List[Hint]

    # creator of State has the option to pass in the precomputed word universe
    # if it is cheaper for the caller to compute it than it would be for this
    # instance of State to compute by scratch. If -O is not passed
    _precomputed_word_universe: t.Optional[t.List[Word]] = None

    def __post_init__(self):
        self._assert_valid_word_universe()

    def _assert_valid_word_universe(self):
        if self._precomputed_word_universe is None:
            return
        words, hints = self.words, self.hints
        assert (
            len(words) == len(hints)
            if self.player == Player.HUMAN
            else len(words) == len(hints) + 1
        ), f"{len(words)=} {len(hints)=}"
        limit = min(len(words), len(hints))
        assert (
            _narrow_word_universe(WORD_LIST, words[:limit], hints[:limit])
            == self._precomputed_word_universe
        )

    @functools.cached_property
    def word_universe(self) -> t.List[Word]:
        if self._precomputed_word_universe is not None:
            return self._precomputed_word_universe
        limit = min(len(self.words), len(self.hints))
        return _narrow_word_universe(
            WORD_LIST, self.words[:limit], self.hints[:limit]
        )

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


WORD_LIST: t.List[Word] = _get_word_list()


def _matches(test_word: Word, hint_word: Word, hint: Hint) -> bool:
    """
    Returns if the word <test_word> is possible given the hint
    (hint_word, hint) was already given
    """
    non_green_letters = [
        c for (c, h) in zip(test_word, hint) if h != HintItem.GREEN
    ]
    for (i, (ch, hint_item)) in enumerate(zip(hint_word, hint)):
        if hint_item == HintItem.GREEN and test_word[i] != ch:
            # print('green')
            return False
        elif hint_item == HintItem.BLACK:
            if ch in non_green_letters:
                return False
        elif hint_item == HintItem.YELLOW:
            # this char must appear in the non green occurances of potword and
            # must not appear in this same index again in potword
            if test_word[i] == ch:
                return False
            if ch not in non_green_letters:
                return False

    return True


# @functools.cache
# def _narrow_wlist(hashable_hint: HashableHint, word: str) -> t.Set[Word]:
#     """ seems to work. Dramatically reduces word list (2000 -> 500). Not good enough """
#     return [
#         output_word
#         for output_word in WORD_LIST
#         if _matches(output_word, word, list(hashable_hint))
#     ]


def _next_possible_words(state: State) -> t.List[Word]:
    """
    actions that HUMAN can take
    given state what are all the possible words that could fit the hints
    """
    assert state.player == Player.HUMAN
    assert len(state.words) == len(state.hints)

    # reduced_word_list = functools.reduce(
    #     lambda a, b: a & b,
    #     (
    #         _narrow_wlist(tuple(hint), word)
    #         for (word, hint) in zip(state.words, state.hints)
    #     ),
    # )
    potential_words = []
    for potential_word in state.word_universe:
        if all(
            _matches(potential_word, guessed_word, hint)
            for guessed_word, hint in zip(state.words, state.hints)
        ):
            potential_words.append(potential_word)
    return potential_words


def _next_possible_hints(state: State) -> t.Iterator[Hint]:
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

    for hint in map(
        list, itertools.product(reversed(HintItem), repeat=WORD_LENGTH)
    ):
        exists = any(
            _matches(word, state.words[-1], hint)
            for word in state.word_universe
        )
        if exists:
            yield hint


# TODO: rename to _format_state
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
    def g(state, *args, **kwargs):
        global _call_stack
        _call_stack += 1
        nodes_visited[_call_stack] += 1
        if _call_stack == 1:
            print(nodes_visited[_call_stack])
            print(_format(state))
        retval = f(state, *args, **kwargs)
        _call_stack -= 1
        return retval

    return g if DEBUG else f


@_debug
def minimax(state: State, alpha=None, beta=None) -> t.List[State]:
    """
    minimax(state, player)

        if player == humman
            return min (minimax(possible next states))
        elif player == absurdle
            return argmmax(board, mi

    speedup with oopening book or look up of words * greens -> {set of possible words}
    returns sequence of states to be taken froom this state under optimal play
    """

    alpha = alpha or float("-inf")  # best option the maximizing player has
    beta = beta or float("+inf")  # best option the minimimizing player has

    if (
        state.is_terminal_node
    ):  # TODO:remove. Degenerate case of next_possible_states
        return [state]
    elif state.player == Player.HUMAN:  # minimizing player

        next_possible_states = (
            State(
                Player.ABSURDLE,
                state.words + [word],
                state.hints,
                state.word_universe,
            )
            for word in _next_possible_words(state)
        )  # TODO: refactor this into general _next_states(s)

        best_val, best_path = float("+inf"), []
        for next_state in next_possible_states:
            path = minimax(next_state, alpha, beta)
            val = len(path[-1].words)
            beta = min(beta, val)
            if beta <= alpha :
                # maximizing player wouldn't have picked this node. This is *not*
                # the proper value of this node but return it anyway cause it's not
                # gonna get picked anyway
                return [state] + path

            if val < best_val:
                best_val = val
                best_path = path
        # If we get here and best_path = [] it means we got a terminal node.
        # Shouldn't be possible if state.is_terminal_node was correct
        assert best_path != []
        result = [state] + best_path

    elif state.player == Player.ABSURDLE:
        # maximizing player
        next_possible_states = (
            State(
                Player.HUMAN,
                state.words,
                state.hints + [hint],
                _narrow_word_universe(
                    state.word_universe, [state.words[-1]], [hint]
                ),
            )
            for hint in _next_possible_hints(state)
        )

        best_val, best_path = float("-inf"), []
        for next_state in next_possible_states:
            path = minimax(next_state, alpha, beta)
            val = len(path[-1].words)
            alpha = max(alpha, val)
            if alpha >= beta:
                # if state.words[-1]== 'dodgy':
                #     import pdb; pdb.set_trace()  # noqa: E702
                # minimizing player wouldn't have picked this node. This is *not*
                # the optimal path from this node but return it anyway cause it's not
                # gonna get picked anyway
                return [state] + path

            if val > best_val:
                best_val = val
                best_path = path
        # If we get here and best_path = [] it means we got a terminal node.
        # Shouldn't be possible if state.is_terminal_node was correct
        assert best_path != []
        result = [state] + best_path
    else:
        raise Exception(
            "Non exhaustive enum (player should be HUMAN or ABSURDLE)"
        )

    # print(f"best result cs={_call_stack}", len(result))
    assert result != [], "Implies no next states"
    return result


def test1():
    start_state = State(Player.ABSURDLE, ["adieu"], [], WORD_LIST)
    print(minimax(start_state))


def test2():

    arr = [
        ("today", list(map(HintItem.from_char, "BBBBB"))),
        ("resin", list(map(HintItem.from_char, "YYBBB"))),
        ("huger", list(map(HintItem.from_char, "BBBYY"))),
        ("creme", list(map(HintItem.from_char, "YGGBB"))),
        ("wreck", list(map(HintItem.from_char, "GGGGG"))),
    ]

    level = 2
    words, hints = zip(*arr[:level])
    words, hints = list(words), list(hints)
    hints = hints[:-1]

    assert len(words) in [len(hints), len(hints) + 1]
    start_state = State(
        Player.ABSURDLE,
        words,
        hints,
        _narrow_word_universe(WORD_LIST, words[:-1], hints),
    )
    state_list = minimax(start_state)
    print(_format(state_list[-1]))


def _get_human_input() -> Word:
    while True:
        inp = input(">>> ENTER WORD :: ").strip()
        if len(inp) != WORD_LENGTH:
            print("Invalid word")
            continue
        return inp.lower()


def _str_to_hint(s: str) -> Hint:
    return [HintItem.from_char(ch.lower()) for ch in s]

def _hint_to_str(hint: Hint) -> str:
    return ''.join(h.char for h in hint)


def _get_absurdle_input() -> Hint:
    while True:
        inp = input(">>> ENTER HINT :: ").strip()
        if len(inp) != WORD_LENGTH:
            print("Invalid hint")
            continue
        try:
            return _str_to_hint(inp)
        except InvalidHintItem:
            print("Invalid hint")
            continue


def interact(player: Player, words: t.List[Word], hints: t.List[Hint]):
    state = State(
        player,
        words,
        hints,
    )
    while not state.is_terminal_node:
        print("CURRENT STATE")
        print(_format(state))
        suggested_next_state = minimax(state)[1]
        suggestion = (
            suggested_next_state.words[-1]
            if state.player == Player.HUMAN
            else "".join(h.char for h in suggested_next_state.hints[-1])
        )
        print("SUGGESTION ::", suggestion)

        if state.player == Player.HUMAN:
            inp = _get_human_input()
            next_state = State(
                Player.ABSURDLE,
                state.words + [inp],
                state.hints,
            )
        elif state.player == Player.ABSURDLE:
            inp = _get_absurdle_input()
            next_state = State(
                Player.HUMAN,
                state.words,
                state.hints + [inp],
            )
        else:
            raise Exception(
                "Non exhaustive enum (player should be HUMAN or ABSURDLE)"
            )

        state = next_state


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
    interact(
        Player.HUMAN,
        ["today", "resin"],
        list(map(_str_to_hint, ["bbbbb", "yybbb"])),
    )
    # test2()
