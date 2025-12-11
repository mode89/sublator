#! vim: ft=paimel

import sublator as sl

def testParseSimpleSingleLine () =
  let content = """
    1
    00:00:01,000 --> 00:00:02,000
    Hello World

    2
    00:00:03,000 --> 00:00:04,000
    Second subtitle
  """ in
  let entries = sl.parseSrt content in (
    assert $ len entries == 2;
    assert $ nth entries 0 == {
      sequenceId: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: "Hello World"
    };
    assert $ nth entries 1 == {
      sequenceId: "2",
      timestamp: "00:00:03,000 --> 00:00:04,000",
      text: "Second subtitle"
    }
  )

def testParseMultiLineSubtitles () =
  let content = """
    1
    00:00:01,000 --> 00:00:02,000
    Line one
    Line two
    Line three

    2
    00:00:03,000 --> 00:00:04,000
    Single line
  """ in
  let entries = sl.parseSrt content in (
    assert $ len entries == 2;
    assert $ nth entries 0 == {
      sequenceId: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: "Line one\nLine two\nLine three"
    };
    assert $ nth entries 1 == {
      sequenceId: "2",
      timestamp: "00:00:03,000 --> 00:00:04,000",
      text: "Single line"
    }
  )

def testParseEmptyContent () = (
  assert $ sl.parseSrt "" |> len |> (==) 0;
  assert $ sl.parseSrt "  \n\n  " |> len |> (==) 0
)

def testParseMalformedEntry () =
  let content = """
    1
    00:00:01,000 --> 00:00:02,000
    Valid entry

    Not a number
    Invalid entry

    3
    00:00:05,000 --> 00:00:06,000
    Another valid entry
  """ in
  let entries = sl.parseSrt content in (
    assert $ len entries == 2;
    assert $ nth entries 0 |. sequenceId |> (==) "1";
    assert $ nth entries 1 |. sequenceId |> (==) "3"
  )

def testParseWithSpecialCharacters () =
  let content = """
    1
    00:00:01,000 --> 00:00:02,000
    [ Sound Effect ]

    2
    00:00:03,000 --> 00:00:04,000
    "Quoted text"

    3
    00:00:05,000 --> 00:00:06,000
    Text with <i>italics</i>
  """ in
  let entries = sl.parseSrt content in (
    assert $ len entries == 3;
    assert $ nth entries 0 |. text |> (==) "[ Sound Effect ]";
    assert $ nth entries 1 |. text |> (==) "\"Quoted text\"";
    assert $ nth entries 2 |. text |> (==) "Text with <i>italics</i>"
  )

def testFormatSimpleEntries () =
  let entries = [
    {
      sequenceId: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: "First subtitle"
    },
    {
      sequenceId: "2",
      timestamp: "00:00:03,000 --> 00:00:04,000",
      text: "Second subtitle"
    }
  ] in
  let output = sl.formatSrt entries in
  let expected = """
    1
    00:00:01,000 --> 00:00:02,000
    First subtitle

    2
    00:00:03,000 --> 00:00:04,000
    Second subtitle

  """ in (
  assert $ output == expected
  )

def testFormatMultiLineSubtitles () =
  let entries = [
    {
      sequenceId: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: "Line one\nLine two"
    }
  ] in
  let output = sl.formatSrt entries in
  assert $ output.find "Line one\nLine two" != -1

def testFormatPreservesTimestampFormat () =
  let timestamp = "00:02:43,747 --> 00:02:47,458" in
  let entries = [{
    sequenceId: "1",
    timestamp: timestamp,
    text: "Text"
  }] in
  let output = sl.formatSrt entries in
  assert $ output.find timestamp != -1

def testFormatEmptyList () =
  assert $ sl.formatSrt [] == ""

def testParseFormatRoundTrip () =
  let original = """
    1
    00:00:01,000 --> 00:00:02,000
    Hello World

    2
    00:00:03,000 --> 00:00:04,000
    Second subtitle

  """ in
  let parsed = sl.parseSrt original in
  let formatted = sl.formatSrt parsed in
  assert $ formatted == original
