#! vim: ft=paimel

import argparse refer { ArgumentParser }

def DEFAULT_MODEL = "google/gemini-2.5-flash-preview-09-2025"
def DEFAULT_BATCH_SIZE = 50

def getParams () = (
  parser := ArgumentParser ();
  parser.add_argument "--language" "-l";
  parser.add_argument "--model" "-m"
    default:DEFAULT_MODEL
    help:"LLM model to use (default: ${DEFAULT_MODEL})";
  parser.add_argument "--batch-size"
    type:int default:DEFAULT_BATCH_SIZE
    help:"""
      Number of subtitles to translate per batch
      (default: ${DEFAULT_BATCH_SIZE})
    """;
  parser.add_argument "--context-size"
    type:int default:nil
    help:"""
      Number of previous translations to include as context
      (default: batch size)
    """;
  let args = parser.parse_args () in {
    language: args.language,
    model: args.model,
  }
)

def parseSrt content =
  let blocks = content
    |. strip ()
    |. split "\n\n"
    |>> filter (fun block ->
      block.split "\n"
      |> len
      |> (>=) 3) in
  let handleBlock block =
    let lines = block.split "\n" in {
      sequenceId: lines.(0).strip (),
      timestamp: lines.(1).strip (),
      text: "\n".join $ drop 2 lines,
    } in
  map handleBlock blocks

def formatSrt entries =
  "\n".join $
  mapcat (fun e -> [e.sequenceId, e.timestamp, e.text, ""]) entries
