#! vim: ft=paimel

import argparse refer { ArgumentParser }
import http.client refer { HTTPException }
import json refer { JSONDecodeError }
import paimel.json as json
import os
import time refer { sleep }
import urllib.request refer { urlopen Request HTTPError URLError }

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

def invokeModel apiKey model prompt =
  let req = Request
    url:"https://openrouter.ai/api/v1/chat/completions"
    method:"POST"
    headers:(
      hashMap
        "Authorization" "Bearer ${apiKey}"
        "Content-Type" "application/json"
    )
    data:(
      {
        model: model,
        messages: [
          {role: "user", content: prompt}
        ],
      }
      |> json.dumps
      |. encode "utf-8"
    )
  in
  loop attempt = 0 in
    if attempt >= 5 then
      raise $ RuntimeError
        "Failed to get response from model after 5 tries."
    else (
      when attempt > 0 do (
        sleep 1.0;
        print $ "Retrying (${attempt}/5)..."
      );
      try
        with res = urlopen req do
          json.loads $ res.read () |. decode "utf-8"
      except HTTPException as e do (
        print $ "HTTP Exception: ${e}\n";
        recur (attempt + 1)
      )
      except HTTPError as e do (
        msg := e.read () |. decode "utf-8";
        print $ "HTTP Error ${e.code}: ${msg}";
        recur (attempt + 1)
      )
      except URLError as e do (
        print $ "URL Error: ${e.reason}";
        recur (attempt + 1)
      )
      except JSONDecodeError as e do (
        print $ "Failed to parse response: ${e}";
        recur (attempt + 1)
      )
    )
