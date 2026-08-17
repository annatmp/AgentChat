from agent_chat.strategies.bidding import Bid, parse_bid


def test_parses_a_well_formed_bid():
    text = "LEVEL: 3\nREASON: the API contract is still undecided"
    assert parse_bid(text) == Bid(level=3, reason="the API contract is still undecided", parsed=True)


def test_accepts_the_boundary_levels():
    assert parse_bid("LEVEL: 0\nREASON: nothing to add").level == 0
    assert parse_bid("LEVEL: 4\nREASON: I was addressed directly").level == 4


def test_is_case_insensitive():
    text = "level: 2\nreason: some general thoughts"
    assert parse_bid(text) == Bid(level=2, reason="some general thoughts", parsed=True)


def test_tolerates_preamble_before_the_level_line():
    # The prompt asks for exactly two lines, but the model doesn't always comply.
    text = "Sure, here is my bid.\nLEVEL: 1\nREASON: nothing urgent"
    assert parse_bid(text) == Bid(level=1, reason="nothing urgent", parsed=True)


def test_missing_level_defaults_to_zero_and_is_flagged_unparsed():
    bid = parse_bid("I don't think I have anything to say right now.")
    assert bid.level == 0
    assert bid.parsed is False


def test_out_of_range_level_is_clamped_and_flagged_unparsed():
    # A malformed level must stay visible as unparsed, not look like a real level-4 bid.
    bid = parse_bid("LEVEL: 7\nREASON: very urgent")
    assert bid.level == 4
    assert bid.parsed is False


def test_missing_reason_defaults_to_a_placeholder():
    bid = parse_bid("LEVEL: 3")
    assert bid.level == 3
    assert bid.reason == "No reason provided."
    assert bid.parsed is True


def test_reason_capture_stops_at_end_of_line():
    text = "LEVEL: 2\nREASON: short reason\nsomething unrelated on the next line"
    assert parse_bid(text).reason == "short reason"
