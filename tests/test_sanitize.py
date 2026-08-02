from agent_chat.sanitize import strip_speaker_echo


def test_strips_single_echoed_tag():
    assert strip_speaker_echo("critic", "[critic]: the plan is thin") == "the plan is thin"


def test_strips_repeated_echoed_tags():
    # The artifact Llama actually produces.
    assert strip_speaker_echo("critic", "[critic]: [critic]: two tags") == "two tags"


def test_is_case_insensitive_and_tolerates_spacing():
    assert strip_speaker_echo("Critic", "  [CRITIC] :  hello") == "hello"


def test_leaves_another_agents_tag_alone():
    # Agents address each other by name; removing this would destroy content.
    text = "[planner]: what do you think?"
    assert strip_speaker_echo("critic", text) == text


def test_leaves_mid_text_tags_alone():
    text = "I agree with [planner]: the estimate is optimistic"
    assert strip_speaker_echo("critic", text) == text


def test_untouched_text_is_returned_verbatim():
    text = "  leading whitespace is preserved when nothing is stripped"
    assert strip_speaker_echo("critic", text) == text


def test_stops_at_the_first_foreign_tag():
    assert strip_speaker_echo("critic", "[critic]: [planner]: hi") == "[planner]: hi"
