#! /usr/bin/env paimel
#! vim: ft=paimel

import os
import sys
import argparse
import builtins refer {FileNotFoundError}
import pathlib refer {Path}
import subprocess
import paimel.json as pj
import urllib.request refer {urlopen, Request}
import urllib.error refer {HTTPError, URLError}
import time refer {sleep}

let DEFAULT_BATCH_SIZE = 100
let DEFAULT_MODEL = "google/gemini-2.5-flash"

let main () =
  let args = buildArgParser () |. parse_args ()
  let exitWithError message =
    print "Error: ${message}" file:sys.stderr
    sys.exit 1
  let validateArgs () = case
    some? args.track_index && nil? args.video ->
      exitWithError "--stream-index requires --video"
    some? args.video && some? args.track_index && nil? args.lang ->
      exitWithError
        "--lang is required when extracting and translating subtitles"
    nil? args.video && nil? args.lang ->
      exitWithError "--lang is required"
  let handleVideoListingMode () =
    when some? args.video && nil? args.track_index do
      try listSubtitleStreams args.video
      except FileNotFoundError, RuntimeError as e do
        exitWithError $ str e
      sys.exit 0
  let requireApiKey () =
    let apiKey = os.getenv "OPENROUTER_API_KEY"
    if apiKey then apiKey
    else exitWithError "OPENROUTER_API_KEY environment variable is not set"
  let readSrtContent () =
    if some? args.video then
      print
        "Extracting subtitles from: ${args.video} (stream ${args.track_index})"
        file:sys.stderr
      try extractSubtitlesFromVideo args.video args.track_index
      except FileNotFoundError, RuntimeError as e do
        exitWithError $ str e
    else
      let srtContent = sys.stdin.read ()
      if srtContent.strip () == ""
      then exitWithError "No input provided via stdin"
      else srtContent
  let parseEntries srtContent =
    let entries = parseSrt srtContent
    if empty? entries
    then exitWithError "No valid subtitle entries found"
    else entries
  let buildContextEntries batchEntries translations contextSize =
    let currentBatchContext =
      batchEntries
        |> mapIndexed (fun i entry -> [entry.text, nth translations i])
        |> vec
    if contextSize == 0 || len currentBatchContext <= contextSize
    then currentBatchContext
    else
      currentBatchContext
        |>> drop (len currentBatchContext - contextSize)
        |> vec
  let batchEntriesAt entries startIndex =
    entries
      |>> drop startIndex
      |>> take args.batch_size
      |> vec
  let printBatchProgress
    batchNumber
    numBatches
    startIndex
    endIndex
    contextEntries =
    let contextCount = len contextEntries
    let baseMessage =
      "Translating batch ${batchNumber}/${numBatches} " +
        "(subtitles ${startIndex + 1}-${endIndex})"
    print (
      if contextCount > 0
      then baseMessage + " with ${contextCount} context entries..."
      else baseMessage + "..."
    ) file:sys.stderr
  let translateCurrentBatch batchEntries contextEntries apiKey contextSize =
    let texts = batchEntries |> map (fun entry -> entry.text) |> vec
    let translations =
      try translateBatch texts args.lang args.model apiKey contextEntries
      except RuntimeError as e do
        print "Error translating batch: ${e}" file:sys.stderr
        sys.exit 1
    {
      translatedBatch:
        batchEntries
          |> mapIndexed (fun i entry -> entry.{text = nth translations i})
          |> vec,
      nextContextEntries:
        buildContextEntries batchEntries translations contextSize,
    }
  let translateEntries entries apiKey contextSize =
    let totalEntries = len entries
    let numBatches =
      (totalEntries + args.batch_size - 1) // args.batch_size
    loop
      startIndex = 0
      batchNumber = 1
      translatedEntries = []
      contextEntries = []
    in
      if startIndex >= totalEntries
      then translatedEntries
      else
        let batchEntries = batchEntriesAt entries startIndex
        let endIndex = startIndex + len batchEntries
        let _ = printBatchProgress
          batchNumber numBatches startIndex endIndex contextEntries
        let result = translateCurrentBatch
          batchEntries contextEntries apiKey contextSize
        let nextTranslatedEntries =
          concat translatedEntries result.translatedBatch |> vec
        recur
          endIndex
          (batchNumber + 1)
          nextTranslatedEntries
          result.nextContextEntries
  validateArgs ()
  handleVideoListingMode ()
  let apiKey = requireApiKey ()
  let contextSize =
    if some? args.context_size then args.context_size else args.batch_size
  let srtContent = readSrtContent ()
  let entries = parseEntries srtContent
  let translatedEntries = translateEntries entries apiKey contextSize
  let output = formatSrt translatedEntries
  print output
  print
    "Translation complete! Processed ${len entries} subtitles."
    file:sys.stderr

let buildArgParser () =
  let parser = argparse.ArgumentParser
    description:"Translate SRT subtitles using LLMs via OpenRouter API"
    epilog:"""
      Examples:
        cat input.srt | sublator.py --lang Spanish > output.srt
        sublator.py --video movie.mkv --lang Spanish > output.srt
    """
  parser.add_argument "-l" "--lang"
    default:nil
    help:"Target language (e.g., Spanish, French, Japanese)"
  parser.add_argument "-m" "--model"
    default:DEFAULT_MODEL
    help:"LLM model to use (default: ${DEFAULT_MODEL})"
  parser.add_argument "--batch-size"
    type:int default:DEFAULT_BATCH_SIZE
    help:"""
      Number of subtitles to translate per batch
      (default: ${DEFAULT_BATCH_SIZE})
    """
  parser.add_argument "--context-size"
    type:int default:nil
    help:"""
      Number of previous translations to include as context
      (default: batch size)
    """
  parser.add_argument "--video"
    type:str default:nil
    help:"""
      Path to video file to extract subtitles from.
      If provided, stdin is ignored.
    """
  parser.add_argument "--stream-index"
    type:int
    dest:"track_index"
    help:"""
      Subtitle stream index to extract from video.
      If not provided with --video, lists available streams and exits.
    """
  parser

let translateBatch texts targetLanguage model apiKey contextEntries:nil =
  let maxRetries = 5
  let expectedCount = len texts
  let batchText =
    texts
      |> mapIndexed (fun i text -> "${i + 1}\n${text}")
      |> "\n---\n".join
  let prompt =
    if notEmpty? contextEntries then
      let previousTranslations =
        contextEntries
          |> map %("${first %}\n===\n${second %}")
          |> "\n---\n".join
      buildTranslatePromptWithContext
        targetLanguage
        previousTranslations:*
        batchText:*
    else buildTranslatePrompt targetLanguage batchText
  let retryOrFail attempt errorMsg =
    if attempt < maxRetries - 1 then
      print
        "Retrying translation (attempt ${attempt + 1}/${maxRetries})..."
        file:sys.stderr
      sleep 1.0
      attempt + 1
    else raise $ RuntimeError $
      "Failed to translate ${expectedCount} entries " +
        "after ${maxRetries} attempts. Last error: ${errorMsg}"
  let orderedTranslations parsedResponse =
    parsedResponse |> sortBy first |> map second |> vec
  loop attempt = 0 in
    let response = invokeModel model prompt apiKey
    try
      let parsedResponse = parseTranslationResponse response expectedCount
      let responseIndices = parsedResponse |> map first |> vec
      let errorMsg = validateIndices expectedCount responseIndices
      if nil? errorMsg
      then orderedTranslations parsedResponse
      else
        print "Warning: Index validation failed: ${errorMsg}" file:sys.stderr
        recur $ retryOrFail attempt errorMsg
    except ValueError as e do
      let errorMsg = str e
      print "Warning: Failed to parse response: ${e}" file:sys.stderr
      recur $ retryOrFail attempt errorMsg

let buildTranslatePrompt targetLanguage batchText = """
  You are a professional subtitle translator. Your task is to translate movie subtitles from their original language into ${targetLanguage}.

  ## Instructions

  1. **Translate each subtitle entry** in the batch below into ${targetLanguage}
  2. **Maintain the exact format**: Each entry should have its index number on the first line and the translated text on subsequent lines
  3. **Separate entries** with `---` exactly as in the input
  4. **Establish consistency**: Since this is the beginning of the movie, pay special attention to how you translate character names, locations, and recurring terms, as these translations will set the standard for subsequent batches

     Example format:
     1
     Translated text here
     ---
     2
     More translated text

  ## Translation Guidelines

  - **Natural and idiomatic**: Translate for meaning and natural flow, not word-for-word
  - **Subtitle constraints**: Keep translations concise and readable within typical subtitle timing
  - **Cultural adaptation**: Adapt idioms, jokes, and cultural references appropriately for the target audience
  - **Character consistency**: Maintain consistent terminology for character names, locations, and recurring phrases
  - **Tone and register**: Preserve the emotional tone, formality level, and speaking style of each character
  - **Technical terms**: Keep proper nouns, brand names, and technical terms consistent with established conventions

  ## Batch to Translate

  ${batchText}

  ## Your Task

  Translate the above subtitle batch into ${targetLanguage} now, maintaining the exact format with index numbers and `---` separators.
"""

let buildTranslatePromptWithContext targetLanguage previousTranslations batchText = """
  You are a professional subtitle translator. Your task is to translate movie subtitles from their original language into ${targetLanguage}.

  ## Instructions

  1. **Translate each subtitle entry** in the batch below into ${targetLanguage}
  2. **Maintain the exact format**: Each entry should have its index number on the first line and the translated text on subsequent lines
  3. **Separate entries** with `---` exactly as in the input
  4. **Ensure continuity**: Use the previous translations provided as context to maintain consistency in terminology, character names, and tone throughout the movie

     Example format:
     1
     Translated text here
     ---
     2
     More translated text

  ## Translation Guidelines

  - **Natural and idiomatic**: Translate for meaning and natural flow, not word-for-word
  - **Subtitle constraints**: Keep translations concise and readable within typical subtitle timing
  - **Cultural adaptation**: Adapt idioms, jokes, and cultural references appropriately for the target audience
  - **Character consistency**: Maintain consistent terminology for character names, locations, and recurring phrases
  - **Tone and register**: Preserve the emotional tone, formality level, and speaking style of each character
  - **Technical terms**: Keep proper nouns, brand names, and technical terms consistent with established conventions

  ## Previous Translations (for context)

  ${previousTranslations}

  ## Current Batch to Translate

  ${batchText}

  ## Your Task

  Translate the above subtitle batch into ${targetLanguage} now, maintaining the exact format with index numbers and `---` separators.
"""

let listSubtitleStreams videoPath =
  let checkFile () =
    when not $ videoPath |> Path |. exists () do
      raise $ FileNotFoundError "Video file not found: ${videoPath}"
  let checkFfprobe () =
    try subprocess.run ["ffprobe", "-version"]
      capture_output:true check:true text:true
    except Exception do
      raise $ RuntimeError
        "ffprobe is not installed or not accessible. \
        Please install ffmpeg to use video subtitle extraction."
  let probeVideo () =
    try subprocess.run
      [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        videoPath,
      ]
      capture_output:true check:true text:true
    except subprocess.CalledProcessError as e do
      let err = if e.stderr then e.stderr else "Unknown error"
      raise $ RuntimeError "Failed to probe video file: ${err}"
  let parseStreams stdout =
    try pj.loads stdout |. get "streams" []
    except Exception as e do
      raise $ RuntimeError "Failed to parse ffprobe output: ${e}"
  let printStream stream =
    let index = stream.get "index" "?"
    let codec = stream.get "codec_name" "unknown"
    let tags = stream.get "tags"
    let lang = if tags then tags.get "language" "unknown" else "unknown"
    let title = if tags then tags.get "title" "" else ""
    print $ if title != ""
      then "  Stream ${index}: ${title} (${lang}, ${codec})"
      else "  Stream ${index}: (${lang}, ${codec})"
  checkFile ()
  checkFfprobe ()
  let subtitleStreams =
    probeVideo ()
      |. stdout
      |> parseStreams
      |> filter %(%.get "codec_type" == "subtitle")
      |> vec
  if len subtitleStreams == 0 then
    print "No subtitle streams found in ${videoPath}" file:sys.stderr
    sys.exit 1
  else
    print "Available subtitle streams in ${videoPath}:"
    for! stream in subtitleStreams do printStream stream

let extractSubtitlesFromVideo videoPath trackIndex =
  let checkFile () =
    when not $ videoPath |> Path |. exists () do
      raise $ FileNotFoundError "Video file not found: ${videoPath}"
  let checkFfmpeg () =
    try subprocess.run ["ffmpeg", "-version"]
      capture_output:true check:true text:true
    except Exception do
      raise $ RuntimeError
        "ffmpeg is not installed or not accessible. \
        Please install ffmpeg to use video subtitle extraction."
  checkFile ()
  checkFfmpeg ()
  try subprocess.run
    ["ffmpeg", "-i", videoPath, "-map", "0:${trackIndex}", "-f", "srt", "-"]
    capture_output:true check:true text:true
    |. stdout
  except subprocess.CalledProcessError as e do
    let err = if e.stderr then e.stderr else "Unknown error"
    raise $ RuntimeError "Failed to extract subtitles from video: ${err}"

let invokeModel model prompt apiKey =
  let maxRetries = 5
  let body = pj.dumps {
    model: model, messages: [{role: "user", content: prompt}]
  } |. encode "utf-8"
  let headers = hashMap
    "Authorization" "Bearer ${apiKey}"
    "Content-Type" "application/json"
  let req = Request "https://openrouter.ai/api/v1/chat/completions"
    method:"POST" headers:headers data:body
  let tryReq () =
    try
      with res = urlopen req do
        res.read () |. decode "utf-8" |> pj.loads
          |. choices |> first |. message |. content
    except HTTPError as e do
      let msg = e.read () |. decode "utf-8"
      print "HTTP Error ${e.code}: ${msg}" file:sys.stderr
    except URLError as e do
      print "URL Error: ${e.reason}" file:sys.stderr
    except Exception as e do
      print "Failed to parse response: ${e}" file:sys.stderr
  loop attempt = 0 in
    let result = tryReq ()
    if some? result then result
    else if attempt < maxRetries - 1 then
      print "Retrying (${attempt + 1}/${maxRetries})..." file:sys.stderr
      sleep 1.0
      recur $ attempt + 1
    else raise $ RuntimeError
      "Failed to get response from model after ${maxRetries} tries."

let parseTranslationResponse response expectedCount =
  let parseBlock i block =
    let block = block.strip ()
    when block != "" do
      let lines = block.split "\n" 1
      let indexLine = lines.(0).strip ()
      let text = if len lines > 1 then lines.(1).strip () else ""
      let index = try int indexLine
        except ValueError as e do
          raise $ ValueError $
            "Entry ${i + 1} has invalid index: '${indexLine}'. " +
              "Expected a number. Error: ${e}"
      when index <= 0 do raise $ ValueError $
        "Entry ${i + 1} has invalid index: ${index}. " +
          "Expected a positive integer (1, 2, 3, ...)"
      [index, text]
  response.split "\n---\n"
    |> mapIndexed parseBlock
    |> filter some?
    |> vec

let validateIndices expectedCount responseIndices =
  let expectedSet = hashSet $* range 1 (expectedCount + 1)
  let responseSet = hashSet $* responseIndices
  case
    missing := seq $ sort $ expectedSet.diff responseSet ->
      "Missing indices: ${missing}"
    extra := seq $ sort $ responseSet.diff expectedSet ->
      "Extra indices found: ${extra}"
    len responseIndices != len responseSet ->
      let duplicates = sort $
        frequencies responseIndices
          |> filter %(%.(1) > 1)
          |> map %(%.(0))
      "Duplicate indices in response: ${duplicates}"

let formatSrt entries =
  if len entries == 0 then ""
  else
    let formatEntry e = "\n".join [e.id, e.timestamp, e.text]
    (entries |> map formatEntry |> "\n\n".join) + "\n\n"

let parseSrt srt =
  let parseBlock b =
    let lines = b.split "\n"
    {
      id: lines.(0).strip (),
      timestamp: lines.(1).strip (),
      text: "\n".join (drop 2 lines)
    }
  srt
    |. strip ()
    |. split "\n\n"
    |> filter %(%.strip () != "")
    |> filter %(len (%.split "\n") >= 3)
    |> map parseBlock
