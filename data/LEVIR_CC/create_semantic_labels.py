import json
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

SEMANTIC_LABELS = [
    "building_added",
    "building_removed",
    "road_added",
    "vegetation_added",
    "vegetation_removed",
    "no_change",
]


OUTPUT_FILE = "semantic_labels.json"
LEVIR_CC_JSON_FILE = "/home/ab6106/Levir-CC-dataset/LevirCCcaptions.json"


def create_prompt(batch):
    """
    Creates a prompt containing a list of filenames and their captions.
    """
    data_str = json.dumps(batch, indent=2)

    return f"""
    You are a data labeling assistant for satellite imagery. 
    Below is a JSON list of image filenames and their 5 associated captions.
    
    For EACH image, analyze the captions and determine if these specific changes occurred.
    Return a JSON object where the key is the filename and the value is an object with these boolean flags:
    - "building_added": true if new buildings/houses/warehouses were constructed.
    - "building_removed": true if buildings were demolished/removed.
    - "road_added": true if new roads/streets/highways were built.
    - "vegetation_added": true if trees/grass were planted.
    - "vegetation_removed": true if trees/forests were cleared or bare land appeared.
    - "no_change": true ONLY if the captions explicitly say "no change", "identical", or "same".

    For some images, multiple changes may have occurred; set all relevant flags to true. If no changes occurred, set only "no_change" to true and all others to false.

    Input Data:
    {data_str}

    Return strict JSON format matching the filenames keys.


    Here is an example.
    Example Input:
    [
      {{
        "filename": "image_001.png",
        "captions": " there is no difference .| the two scenes seem identical .| the scene is the same as before .| no change has occurred .| almost nothing has changed ."
      }},
      {{
        "filename": "image_002.png",
        "captions": " a row of houses with a swimming pool appears at the bottom left corner of the scene .| some detached houses appear beside the trees .| a row of buildings has been constructed .| several buildings are built on the lower-left . a concrete and some houses are constructed in the clearing ."
      }}
    ]

    Example Output:
    {{
      "image_001.png": {{
        "building_added": false,
        "building_removed": false,
        "road_added": false,
        "vegetation_added": false,
        "vegetation_removed": false,
        "no_change": true
      }},
      "image_002.png": {{
        "building_added": true,
        "building_removed": false,
        "road_added": false,
        "vegetation_added": false,
        "vegetation_removed": false,
        "no_change": false
      }}
    }}
    """


def main():
    with open(LEVIR_CC_JSON_FILE, "r") as f:
        data = json.load(f)
    images = data["images"]
    images_with_captions = []
    for img in images:
        filename = img["filename"]
        concatenated_captions = "|".join(
            sentence["raw"] for sentence in img["sentences"]
        )
        images_with_captions.append(
            {"filename": filename, "captions": concatenated_captions}
        )

    BATCH_SIZE = 50
    batches = [
        images_with_captions[i : i + BATCH_SIZE]
        for i in range(0, len(images_with_captions), BATCH_SIZE)
    ]

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
        print(f"Resuming... Found {len(results)} existing labels.")
    else:
        results = {}

    print(f"Starting processing of {len(batches)} batches...")

    for i, batch in tqdm(enumerate(batches)):
        # Skip if all items in this batch are already done
        if all(item["filename"] in results for item in batch):
            print(f"Skipping batch {i}, all items already processed.")
            continue

        try:
            print()
            prompt = create_prompt(batch)
            print(f"Prompt created for batch {i}.")

            # Call API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            print(f"Response received for batch {i}.")

            # Parse Response
            batch_result = json.loads(response.text)

            if i % 10 == 0:
                print(f"Processed {i+1} / {len(batches)} batches.")
                print(json.dumps(batch_result, indent=2))

            # Update main dictionary
            results.update(batch_result)

            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error on batch: {e}")
            time.sleep(10)  # Cool down on error


if __name__ == "__main__":
    main()
