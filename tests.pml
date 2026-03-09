#! vim: ft=paimel

import sublator as sl
import paimel.json as pj
import builtins as py
import pytest
import subprocess
import unittest.mock refer {patch, MagicMock}
import urllib.error refer {URLError}

# SRT Parsing Tests

let testParseSrtSingleLine () =
  let srt = """
    1
    00:00:01,000 --> 00:00:02,000
    Hello World

    2
    00:00:03,000 --> 00:00:04,000
    Second subtitle
  """
  let entries = sl.parseSrt srt
  assert $ len entries == 2
  assert $ nth entries 0 == {
    id: "1",
    timestamp: "00:00:01,000 --> 00:00:02,000",
    text: "Hello World"
  }
  assert $ nth entries 1 == {
    id: "2",
    timestamp: "00:00:03,000 --> 00:00:04,000",
    text: "Second subtitle"
  }

let testParseSrtMultilineSubtitles () =
  let entries = sl.parseSrt """
    1
    00:00:01,000 --> 00:00:02,000
    Line one
    Line two
    Line three

    2
    00:00:03,000 --> 00:00:04,000
    Single line
  """
  assert $ len entries == 2
  assert $ (nth entries 0).text == """
    Line one
    Line two
    Line three
  """
  assert $ (nth entries 1).text == "Single line"

let testParseSrtSpecialCharacters () =
  let entries = sl.parseSrt """
    1
    00:00:01,000 --> 00:00:02,000
    [ Sound Effect ]

    2
    00:00:03,000 --> 00:00:04,000
    "Quoted text"

    3
    00:00:05,000 --> 00:00:06,000
    Text with <i>italics</i>
  """
  assert $ len entries == 3
  assert $ (nth entries 0).text == "[ Sound Effect ]"
  assert $ (nth entries 1).text == "\"Quoted text\""
  assert $ (nth entries 2).text == "Text with <i>italics</i>"

let testParseSrtEmptyContent () =
  assert $ len (sl.parseSrt "") == 0
  assert $ len (sl.parseSrt "   \n\n  ") == 0

# SRT Formatting Tests

let testFormatSrtSimpleEntries () =
  let entries = [
    {
      id: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: "First subtitle"
    },
    {
      id: "2",
      timestamp: "00:00:03,000 --> 00:00:04,000",
      text: "Second subtitle"
    },
  ]
  let output = sl.formatSrt entries
  let expected = """
    1
    00:00:01,000 --> 00:00:02,000
    First subtitle

    2
    00:00:03,000 --> 00:00:04,000
    Second subtitle


  """
  assert $ output == expected

let testFormatSrtMultilineSubtitles () =
  let output = sl.formatSrt [
    {
      id: "1",
      timestamp: "00:00:01,000 --> 00:00:02,000",
      text: """
        Line one
        Line two
      """,
    },
  ]
  assert $ output == """
    1
    00:00:01,000 --> 00:00:02,000
    Line one
    Line two


  """

let testFormatSrtTimestampFormat () =
  let output = sl.formatSrt [
    {id: "1", timestamp: "00:02:43,747 --> 00:02:47,458", text: "Text"},
  ]
  assert $ output == """
    1
    00:02:43,747 --> 00:02:47,458
    Text


  """

let testFormatSrtEmptyList () =
  assert $ sl.formatSrt [] == ""

# Parse Translation Response Tests

let testParseTranslationValidResponse () =
  let parsed = sl.parseTranslationResponse """
    1
    Translated one
    ---
    2
    Translated two
    ---
    3
    Translated three
  """ 3
  assert $ len parsed == 3
  assert $ parsed.(0) == [1, "Translated one"]
  assert $ parsed.(1) == [2, "Translated two"]
  assert $ parsed.(2) == [3, "Translated three"]

let testParseTranslationMultilineText () =
  let parsed = sl.parseTranslationResponse """
    1
    Line 1
    Line 2
    ---
    2
    Single line
  """ 2
  assert $ len parsed == 2
  assert $ parsed.(0) == [1, "Line 1\nLine 2"]
  assert $ parsed.(1) == [2, "Single line"]

let testParseTranslationInvalidIndex () =
  let result = try sl.parseTranslationResponse """
    abc
    Translated text
  """ 1
    except ValueError as e do str e
  assert $ result.count "invalid index"

let testParseTranslationNegativeIndex () =
  let result = try sl.parseTranslationResponse """
    -1
    Translated text
  """ 1
    except ValueError as e do str e
  assert $ result.count "invalid index: -1"

let testParseTranslationZeroIndex () =
  let result = try sl.parseTranslationResponse """
    0
    Translated text
  """ 1
    except ValueError as e do str e
  assert $ result.count "invalid index: 0"

let testParseTranslationMissingIndex () =
  let result =
    try sl.parseTranslationResponse
      "Just translated text without index" 1
    except ValueError as e do str e
  assert $ result.count "invalid index"

let testParseTranslationEmptyText () =
  let parsed = sl.parseTranslationResponse """
    1
    ---
    2
    Some text
  """ 2
  assert $ len parsed == 2
  assert $ parsed.(0) == [1, ""]
  assert $ parsed.(1) == [2, "Some text"]

# Index Validation Tests

let testValidateIndicesCompleteSequence () =
  assert $ sl.validateIndices 3 [1, 2, 3] is nil

let testValidateIndicesOutOfOrder () =
  assert $ sl.validateIndices 3 [3, 1, 2] is nil

let testValidateIndicesMissing () =
  let result = sl.validateIndices 4 [1, 2, 4]
  assert $ result.count "Missing indices"
  assert $ result.count "3"

let testValidateIndicesExtra () =
  let result = sl.validateIndices 2 [1, 2, 3]
  assert $ result.count "Extra indices"
  assert $ result.count "3"

let testValidateIndicesDuplicates () =
  let result = sl.validateIndices 3 [1, 2, 2, 3]
  assert $ result.count "Duplicate indices"
  assert $ result.count "2"

let testValidateIndicesMissingMultiple () =
  let result = sl.validateIndices 5 [1, 3, 5]
  assert $ result.count "Missing indices"
  assert $ result.count "2"
  assert $ result.count "4"

# Batch Translation Tests

let testTranslateBatchSuccess () =
  with invokeModel = patch "sublator.invokeModel"
    return_value:"""
      1
      Spanish 1
      ---
      2
      Spanish 2
      ---
      3
      Spanish 3
    """
  do
    let translations =
      sl.translateBatch
        ["English 1", "English 2", "English 3"]
        "Spanish"
        "test-model"
        "test-key"
    let callArgs = invokeModel.call_args.(0)
    assert $ translations == ["Spanish 1", "Spanish 2", "Spanish 3"]
    invokeModel.assert_called_once ()
    assert $ nth callArgs 0 == "test-model"
    assert $ (nth callArgs 1).count "Spanish"
    assert $ nth callArgs 2 == "test-key"

let testTranslateBatchIndexMismatch () =
  let effects = py.list [
    """
      1
      Spanish 1
      ---
      3
      Spanish 3
    """,
    """
      1
      Spanish 1
      ---
      2
      Spanish 2
      ---
      3
      Spanish 3
    """,
  ]
  with
    sleep = patch "sublator.sleep"
    invokeModel = patch "sublator.invokeModel" side_effect:effects
  do
    let translations =
      sl.translateBatch
        ["English 1", "English 2", "English 3"]
        "Spanish"
        "test-model"
        "test-key"
    assert $ translations == ["Spanish 1", "Spanish 2", "Spanish 3"]
    assert $ invokeModel.call_count == 2
    assert $ sleep.call_count == 1

let testTranslateBatchMultilineText () =
  with invokeModel = patch "sublator.invokeModel"
    return_value:"""
      1
      Spanish line 1
      Spanish line 2
    """
  do
    let translations = sl.translateBatch
      ["""
        English line 1
        English line 2
      """]
      "Spanish"
      "test-model"
      "test-key"
    assert $ len translations == 1
    assert $ translations.(0).count "\n"

let testTranslateBatchWithContext () =
  with invokeModel = patch "sublator.invokeModel"
    return_value:"""
      1
      Translated 1
      ---
      2
      Translated 2
    """
  do
    let contextEntries = [["Prev 1", "Prev T1"], ["Prev 2", "Prev T2"]]
    let translations = sl.translateBatch
      ["English 1", "English 2"]
      "Spanish"
      "test-model"
      "test-key"
      contextEntries
    let prompt = invokeModel.call_args.(0).(1)
    assert $ translations == ["Translated 1", "Translated 2"]
    assert $ prompt.count """
      Prev 1
      ===
      Prev T1
    """
    assert $ prompt.count """
      Prev 2
      ===
      Prev T2
    """

let testTranslateBatchWithoutContext () =
  with invokeModel = patch "sublator.invokeModel"
    return_value:"""
      1
      Spanish 1
      ---
      2
      Spanish 2
    """
  do
    let translations = sl.translateBatch
      ["English 1", "English 2"]
      "Spanish"
      "test-model"
      "test-key"
    let prompt = invokeModel.call_args.(0).(1)
    assert $ translations == ["Spanish 1", "Spanish 2"]
    assert $ prompt.count "Previous" == 0
    assert $ prompt.count "===" == 0

let testTranslateBatchEmptyContext () =
  with invokeModel = patch "sublator.invokeModel"
    return_value:"""
      1
      Spanish 1
      ---
      2
      Spanish 2
    """
  do
    let _ = sl.translateBatch
      ["English 1", "English 2"]
      "Spanish"
      "test-model"
      "test-key"
      []
    let prompt = invokeModel.call_args.(0).(1)
    assert $ (prompt |. lower () |. count "context") == 0

let testTranslateBatchMaxRetriesExceeded () =
  with
    sleep = patch "sublator.sleep"
    invokeModel = patch "sublator.invokeModel"
      return_value:"""
        1
        Only one translation
      """
  do
    with _ = pytest.raises RuntimeError
      match:"Failed to translate 2 entries after 5 attempts"
    do
      sl.translateBatch
        ["English 1", "English 2"]
        "Spanish"
        "test-model"
        "test-key"
    assert $ invokeModel.call_count == 5
    assert $ sleep.call_count == 4

# API Invocation Tests

let testInvokeModelSuccess () =
  let resBody = pj.dumps {choices: [{message: {content: "Translated text"}}]}
    |. encode "utf-8"
  let res = MagicMock read:(MagicMock return_value:resBody)
  let resCtx = MagicMock __enter__:(MagicMock return_value:res)
  with urlopen = patch "sublator.urlopen" return_value:resCtx do
    assert $ sl.invokeModel "test-model" "Test prompt" "test-key"
      == "Translated text"
    urlopen.assert_called_once ()

let testInvokeModelRetryError () =
  let resBody = pj.dumps {choices: [{message: {content: "Success"}}]}
    |. encode "utf-8"
  let res = MagicMock read:(MagicMock return_value:resBody)
  let resCtx = MagicMock __enter__:(MagicMock return_value:res)
  let effects = py.list [
    URLError "Connection error",
    URLError "Connection error",
    resCtx
  ]
  with
    sleep = patch "sublator.sleep"
    urlopen = patch "sublator.urlopen" side_effect:effects
  do
    assert $ sl.invokeModel "test-model" "Test prompt" "test-key"
      == "Success"
    assert $ urlopen.call_count == 3
    assert $ sleep.call_count == 2

let testInvokeModelMaxRetriesExceeded () =
  with
    sleep = patch "sublator.sleep"
    urlopen = patch "sublator.urlopen"
      side_effect:(URLError "Connection error")
  do
    with _ = pytest.raises RuntimeError match:"after 5 tries" do
      sl.invokeModel "test-model" "Test prompt" "test-key"
    assert $ urlopen.call_count == 5
    assert $ sleep.call_count == 4

# Video Extraction Tests

let testExtractSubtitlesSuccess () =
  let versionResult = MagicMock returncode:0 stdout:"ffmpeg version 6.0.0"
  let extractResult = MagicMock returncode:0 stderr:"" stdout:"""
    1
    00:00:01,000 --> 00:00:02,000
    Extracted subtitle

    2
    00:00:03,000 --> 00:00:04,000
    Second extracted subtitle
  """
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, extractResult])
    exists = patch "sublator.Path.exists" return_value:true
  do
    let result = sl.extractSubtitlesFromVideo "test.mp4" 0
    assert $ result.count "Extracted subtitle"
    assert $ result.count "Second extracted subtitle"
    assert $ run.call_count == 2
    let versionCall = run.call_args_list.(0).(0).(0)
    let extractCall = run.call_args_list.(1).(0).(0)
    assert $ versionCall == py.list ["ffmpeg", "-version"]
    assert $ nth extractCall 0 == "ffmpeg"
    assert $ nth extractCall 1 == "-i"
    assert $ nth extractCall 2 == "test.mp4"
    assert $ contains? extractCall "0:0"

let testExtractSubtitlesCustomTrack () =
  let versionResult = MagicMock returncode:0 stdout:"ffmpeg version 6.0.0"
  let extractResult = MagicMock returncode:0 stderr:"" stdout:"""
    1
    00:00:01,000 --> 00:00:02,000
    Subtitle
  """
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, extractResult])
    exists = patch "sublator.Path.exists" return_value:true
  do
    sl.extractSubtitlesFromVideo "test.mkv" 2
    assert $ contains? run.call_args_list.(1).(0).(0) "0:2"

let testExtractSubtitlesFileNotFound () =
  with _ = pytest.raises py.FileNotFoundError match:"Video file not found" do
    sl.extractSubtitlesFromVideo "nonexistent.mp4" 0

let testExtractSubtitlesFfmpegNotFound () =
  with
    run = patch "subprocess.run"
      side_effect:(py.FileNotFoundError "ffmpeg not found")
    exists = patch "sublator.Path.exists" return_value:true
    _ = pytest.raises RuntimeError match:"ffmpeg is not installed"
  do
    sl.extractSubtitlesFromVideo "test.mp4" 0

let testExtractSubtitlesFfmpegFailure () =
  let versionResult = MagicMock returncode:0 stdout:"ffmpeg version 6.0.0"
  let error = subprocess.CalledProcessError 1 "ffmpeg"
  let _ = set! error.stderr "Invalid data when processing input"
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, error])
    exists = patch "sublator.Path.exists" return_value:true
    _ = pytest.raises RuntimeError match:"Failed to extract subtitles"
  do
    sl.extractSubtitlesFromVideo "test.mp4" 0

let testExtractSubtitlesNoSubtitleStream () =
  let versionResult = MagicMock returncode:0 stdout:"ffmpeg version 6.0.0"
  let error = subprocess.CalledProcessError 1 "ffmpeg"
  let _ = set! error.stderr "Stream #0:0: not found"
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, error])
    exists = patch "sublator.Path.exists" return_value:true
    _ = pytest.raises RuntimeError match:"Failed to extract subtitles"
  do
    sl.extractSubtitlesFromVideo "no_subs.mp4" 0

let testListStreamsSuccess capsys =
  let versionResult = MagicMock returncode:0 stdout:"ffprobe version 6.0.0"
  let streamsResult = MagicMock returncode:0 stdout:(pj.dumps {
    streams: [
      {index: 0, codec_type: "video", codec_name: "h264"},
      {index: 1, codec_type: "audio", codec_name: "aac"},
      {
        index: 2,
        codec_type: "subtitle",
        codec_name: "subrip",
        tags: {language: "eng", title: "English Subtitles"},
      },
      {
        index: 3,
        codec_type: "subtitle",
        codec_name: "subrip",
        tags: {language: "spa", title: "Spanish Subtitles"},
      },
    ],
  })
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, streamsResult])
    exists = patch "sublator.Path.exists" return_value:true
  do
    sl.listSubtitleStreams "test.mkv"
    let captured = capsys.readouterr ()
    let versionCall = run.call_args_list.(0).(0).(0)
    let probeCall = run.call_args_list.(1).(0).(0)
    assert $ versionCall == py.list ["ffprobe", "-version"]
    assert $ nth probeCall 0 == "ffprobe"
    assert $ contains? probeCall "-show_streams"
    assert $ contains? probeCall "test.mkv"
    assert $ captured.out.count "Available subtitle streams in test.mkv:"
    assert $ captured.out.count "Stream 2: English Subtitles (eng, subrip)"
    assert $ captured.out.count "Stream 3: Spanish Subtitles (spa, subrip)"

let testListStreamsNoSubtitles capsys =
  let versionResult = MagicMock returncode:0 stdout:"ffprobe version 6.0.0"
  let streamsResult = MagicMock returncode:0 stdout:(pj.dumps {
    streams: [
      {index: 0, codec_type: "video", codec_name: "h264"},
      {index: 1, codec_type: "audio", codec_name: "aac"},
    ],
  })
  with
    run = patch "subprocess.run"
      side_effect:(py.list [versionResult, streamsResult])
    exists = patch "sublator.Path.exists" return_value:true
  do
    with _ = pytest.raises py.SystemExit do
      sl.listSubtitleStreams "no_subs.mkv"
    let captured = capsys.readouterr ()
    assert $ captured.err.count "No subtitle streams found in no_subs.mkv"

let testListStreamsFileNotFound () =
  with _ = pytest.raises py.FileNotFoundError match:"Video file not found" do
    sl.listSubtitleStreams "nonexistent.mkv"

let testListStreamsFfprobeNotFound () =
  with
    run = patch "subprocess.run"
      side_effect:(py.FileNotFoundError "ffprobe not found")
    exists = patch "sublator.Path.exists" return_value:true
    _ = pytest.raises RuntimeError match:"ffprobe is not installed"
  do
    sl.listSubtitleStreams "test.mkv"

# CLI Argument Tests

let testArgParserDefaultModel () =
  let parser = sl.buildArgParser ()
  let args = parser.parse_args $ py.list ["--lang", "Spanish"]
  assert $ args.model == "google/gemini-2.5-flash"
  assert $ args.batch_size == 100

let testArgParserCustomModelBatchSize () =
  let parser = sl.buildArgParser ()
  let args = parser.parse_args $ py.list [
    "--lang", "French",
    "--model", "custom-model",
    "--batch-size", "50",
  ]
  assert $ args.lang == "French"
  assert $ args.model == "custom-model"
  assert $ args.batch_size == 50

let testArgParserVideoDefaults () =
  let parser = sl.buildArgParser ()
  let args = parser.parse_args $ py.list ["--lang", "Spanish"]
  assert $ args.video is nil
  assert $ args.track_index is nil

let testArgParserCustomTrackIndex () =
  let parser = sl.buildArgParser ()
  let args = parser.parse_args $ py.list [
    "--lang", "Spanish",
    "--video", "movie.mkv",
    "--stream-index", "2",
  ]
  assert $ args.video == "movie.mkv"
  assert $ args.track_index == 2

let testArgParserTrackIndexOnly () =
  let parser = sl.buildArgParser ()
  let args = parser.parse_args $ py.list [
    "--lang", "Spanish",
    "--stream-index", "3",
  ]
  assert $ args.video is nil
  assert $ args.track_index == 3

let makeMainParser
  lang:"Spanish"
  model:"test-model"
  batchSize:2
  contextSize:nil
  video:nil
  trackIndex:nil =
  let args = MagicMock
    lang:lang
    model:model
    batch_size:batchSize
    context_size:contextSize
    video:video
    track_index:trackIndex
  MagicMock parse_args:(MagicMock return_value:args)

let testMainRejectsTrackIndexWithoutVideo capsys =
  let parser = makeMainParser trackIndex:1
  with buildArgParser = patch "sublator.buildArgParser" return_value:parser do
    let exitCode = try sl.main ()
      except py.SystemExit as e do e.code
    let captured = capsys.readouterr ()
    assert $ exitCode == 1
    assert $ captured.err.count "--stream-index requires --video"

let testMainListsStreamsVideoOnly () =
  let parser = makeMainParser lang:nil video:"movie.mkv"
  with
    buildArgParser = patch "sublator.buildArgParser" return_value:parser
    listSubtitleStreams = patch "sublator.listSubtitleStreams"
  do
    let exitCode = try sl.main ()
      except py.SystemExit as e do e.code
    assert $ exitCode == 0
    listSubtitleStreams.assert_called_once ()
    assert $ listSubtitleStreams.call_args.(0).(0) == "movie.mkv"

let testMainRejectsMissingLangVideoTranslate capsys =
  let parser = makeMainParser lang:nil video:"movie.mkv" trackIndex:2
  with buildArgParser = patch "sublator.buildArgParser" return_value:parser do
    let exitCode = try sl.main ()
      except py.SystemExit as e do e.code
    let captured = capsys.readouterr ()
    assert $ exitCode == 1
    assert $ captured.err.count
      "--lang is required when extracting and translating subtitles"

let testMainRejectsMissingApiKey capsys =
  let parser = makeMainParser ()
  with
    buildArgParser = patch "sublator.buildArgParser" return_value:parser
    getenv = patch "os.getenv" return_value:nil
  do
    let exitCode = try sl.main ()
      except py.SystemExit as e do e.code
    let captured = capsys.readouterr ()
    assert $ exitCode == 1
    assert $ captured.err.count
      "OPENROUTER_API_KEY environment variable is not set"

let testMainTranslatesStdinSingleBatch capsys =
  let parser = makeMainParser batchSize:10
  let srt = """
    1
    00:00:01,000 --> 00:00:02,000
    Hello world
  """
  with
    buildArgParser = patch "sublator.buildArgParser" return_value:parser
    getenv = patch "os.getenv" return_value:"test-key"
    read = patch "sys.stdin.read" return_value:srt
    translateBatch = patch "sublator.translateBatch"
      return_value:["Hola mundo"]
  do
    sl.main ()
    let captured = capsys.readouterr ()
    let callArgs = translateBatch.call_args.(0)
    translateBatch.assert_called_once ()
    assert $ len callArgs.(0) == 1
    assert $ callArgs.(0).(0) == "Hello world"
    assert $ callArgs.(1) == "Spanish"
    assert $ callArgs.(2) == "test-model"
    assert $ callArgs.(3) == "test-key"
    assert $ len callArgs.(4) == 0
    assert $ captured.out.count "Hola mundo"
    assert $ captured.err.count "Translating batch 1/1"
    assert $ captured.err.count
      "Translation complete! Processed 1 subtitles."

let testMainUsesBatchSizeDefaultContext capsys =
  let parser = makeMainParser batchSize:2
  let srt = """
    1
    00:00:01,000 --> 00:00:02,000
    One

    2
    00:00:03,000 --> 00:00:04,000
    Two

    3
    00:00:05,000 --> 00:00:06,000
    Three
  """
  with
    buildArgParser = patch "sublator.buildArgParser" return_value:parser
    getenv = patch "os.getenv" return_value:"test-key"
    read = patch "sys.stdin.read" return_value:srt
    translateBatch = patch "sublator.translateBatch"
      side_effect:(py.list [["Uno", "Dos"], ["Tres"]])
  do
    sl.main ()
    let secondCallArgs = translateBatch.call_args_list.(1).(0)
    let contextEntries = secondCallArgs.(4)
    let captured = capsys.readouterr ()
    assert $ translateBatch.call_count == 2
    assert $ len contextEntries == 2
    assert $ contextEntries.(0) == ["One", "Uno"]
    assert $ contextEntries.(1) == ["Two", "Dos"]
    assert $ len secondCallArgs.(0) == 1
    assert $ secondCallArgs.(0).(0) == "Three"
    assert $ captured.err.count "with 2 context entries"
